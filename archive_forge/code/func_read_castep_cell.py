import os
import re
import warnings
import numpy as np
from copy import deepcopy
import ase
from ase.parallel import paropen
from ase.spacegroup import Spacegroup
from ase.geometry.cell import cellpar_to_cell
from ase.constraints import FixAtoms, FixedPlane, FixedLine, FixCartesian
from ase.utils import atoms_to_spglib_cell
import ase.units
def read_castep_cell(fd, index=None, calculator_args={}, find_spg=False, units=units_CODATA2002):
    """Read a .cell file and return an atoms object.
    Any value found that does not fit the atoms API
    will be stored in the atoms.calc attribute.

    By default, the Castep calculator will be tolerant and in the absence of a
    castep_keywords.json file it will just accept all keywords that aren't
    automatically parsed.
    """
    from ase.calculators.castep import Castep
    cell_units = {'bohr': units_CODATA2002['a0'], 'ang': 1.0, 'm': 10000000000.0, 'cm': 100000000.0, 'nm': 10, 'pm': 0.01}
    calc = Castep(**calculator_args)
    if calc.cell.castep_version == 0 and calc._kw_tol < 3:
        warnings.warn('read_cell: Warning - Was not able to validate CASTEP input. This may be due to a non-existing "castep_keywords.json" file or a non-existing CASTEP installation. Parsing will go on but keywords will not be validated and may cause problems if incorrect during a CASTEP run.')
    celldict = read_freeform(fd)

    def parse_blockunit(line_tokens, blockname):
        u = 1.0
        if len(line_tokens[0]) == 1:
            usymb = line_tokens[0][0].lower()
            u = cell_units.get(usymb, 1)
            if usymb not in cell_units:
                warnings.warn('read_cell: Warning - ignoring invalid unit specifier in %BLOCK {0} (assuming Angstrom instead)'.format(blockname))
            line_tokens = line_tokens[1:]
        return (u, line_tokens)
    aargs = {'pbc': True}
    lat_keywords = [w in celldict for w in ('lattice_cart', 'lattice_abc')]
    if all(lat_keywords):
        warnings.warn('read_cell: Warning - two lattice blocks present in the same file. LATTICE_ABC will be ignored')
    elif not any(lat_keywords):
        raise ValueError('Cell file must contain at least one between LATTICE_ABC and LATTICE_CART')
    if 'lattice_abc' in celldict:
        lines = celldict.pop('lattice_abc')[0].split('\n')
        line_tokens = [l.split() for l in lines]
        u, line_tokens = parse_blockunit(line_tokens, 'lattice_abc')
        if len(line_tokens) != 2:
            warnings.warn('read_cell: Warning - ignoring additional lines in invalid %BLOCK LATTICE_ABC')
        abc = [float(p) * u for p in line_tokens[0][:3]]
        angles = [float(phi) for phi in line_tokens[1][:3]]
        aargs['cell'] = cellpar_to_cell(abc + angles)
    if 'lattice_cart' in celldict:
        lines = celldict.pop('lattice_cart')[0].split('\n')
        line_tokens = [l.split() for l in lines]
        u, line_tokens = parse_blockunit(line_tokens, 'lattice_cart')
        if len(line_tokens) != 3:
            warnings.warn('read_cell: Warning - ignoring more than three lattice vectors in invalid %BLOCK LATTICE_CART')
        aargs['cell'] = [[float(x) * u for x in lt[:3]] for lt in line_tokens]
    pos_keywords = [w in celldict for w in ('positions_abs', 'positions_frac')]
    if all(pos_keywords):
        warnings.warn('read_cell: Warning - two lattice blocks present in the same file. POSITIONS_FRAC will be ignored')
        del celldict['positions_frac']
    elif not any(pos_keywords):
        raise ValueError('Cell file must contain at least one between POSITIONS_FRAC and POSITIONS_ABS')
    aargs['symbols'] = []
    pos_type = 'positions'
    pos_block = celldict.pop('positions_abs', [None])[0]
    if pos_block is None:
        pos_type = 'scaled_positions'
        pos_block = celldict.pop('positions_frac', [None])[0]
    aargs[pos_type] = []
    lines = pos_block.split('\n')
    line_tokens = [l.split() for l in lines]
    if 'scaled' not in pos_type:
        u, line_tokens = parse_blockunit(line_tokens, 'positions_abs')
    else:
        u = 1.0
    add_info = {'SPIN': (float, 0.0), 'MAGMOM': (float, 0.0), 'LABEL': (str, 'NULL')}
    add_info_arrays = dict(((k, []) for k in add_info))

    def parse_info(raw_info):
        re_keys = '({0})\\s*[=:\\s]{{1}}\\s*([^\\s]*)'.format('|'.join(add_info.keys()))
        info = re.findall(re_keys, raw_info)
        info = {g[0]: add_info[g[0]][0](g[1]) for g in info}
        return info
    custom_species = None
    for tokens in line_tokens:
        spec_custom = tokens[0].split(':', 1)
        elem = spec_custom[0]
        if len(spec_custom) > 1 and custom_species is None:
            custom_species = list(aargs['symbols'])
        if custom_species is not None:
            custom_species.append(tokens[0])
        aargs['symbols'].append(elem)
        aargs[pos_type].append([float(p) * u for p in tokens[1:4]])
        info = ' '.join(tokens[4:])
        info = parse_info(info)
        for k in add_info:
            add_info_arrays[k] += [info.get(k, add_info[k][1])]
    if 'species_pot' in celldict:
        lines = celldict.pop('species_pot')[0].split('\n')
        line_tokens = [l.split() for l in lines]
        for tokens in line_tokens:
            if len(tokens) == 1:
                all_spec = set(custom_species) if custom_species is not None else set(aargs['symbols'])
                for s in all_spec:
                    calc.cell.species_pot = (s, tokens[0])
            else:
                calc.cell.species_pot = tuple(tokens[:2])
    raw_constraints = {}
    if 'ionic_constraints' in celldict:
        lines = celldict.pop('ionic_constraints')[0].split('\n')
        line_tokens = [l.split() for l in lines]
        for tokens in line_tokens:
            if not len(tokens) == 6:
                continue
            _, species, nic, x, y, z = tokens
            x = float(x)
            y = float(y)
            z = float(z)
            nic = int(nic)
            if (species, nic) not in raw_constraints:
                raw_constraints[species, nic] = []
            raw_constraints[species, nic].append(np.array([x, y, z]))
    if 'symmetry_ops' in celldict:
        lines = celldict.pop('symmetry_ops')[0].split('\n')
        line_tokens = [l.split() for l in lines]
        blocks = np.array(line_tokens).astype(float)
        if len(blocks.shape) != 2 or blocks.shape[1] != 3 or blocks.shape[0] % 4 != 0:
            warnings.warn('Warning: could not parse SYMMETRY_OPS block properly, skipping')
        else:
            blocks = blocks.reshape((-1, 4, 3))
            rotations = blocks[:, :3]
            translations = blocks[:, 3]
            calc.cell.symmetry_ops = (rotations, translations)
    for k, (val, otype) in celldict.items():
        try:
            if otype == 'block':
                val = val.split('\n')
            calc.cell.__setattr__(k, val)
        except Exception as e:
            raise RuntimeError('Problem setting calc.cell.%s = %s: %s' % (k, val, e))
    aargs['magmoms'] = np.array(add_info_arrays['SPIN'])
    aargs['magmoms'] = np.where(aargs['magmoms'] != 0, aargs['magmoms'], add_info_arrays['MAGMOM'])
    labels = np.array(add_info_arrays['LABEL'])
    aargs['calculator'] = calc
    atoms = ase.Atoms(**aargs)
    if find_spg:
        try:
            import spglib
        except ImportError:
            warnings.warn('spglib not found installed on this system - automatic spacegroup detection is not possible')
            spglib = None
        if spglib is not None:
            symmd = spglib.get_symmetry_dataset(atoms_to_spglib_cell(atoms))
            atoms_spg = Spacegroup(int(symmd['number']))
            atoms.info['spacegroup'] = atoms_spg
    atoms.new_array('castep_labels', labels)
    if custom_species is not None:
        atoms.new_array('castep_custom_species', np.array(custom_species))
    fixed_atoms = []
    constraints = []
    for (species, nic), value in raw_constraints.items():
        absolute_nr = atoms.calc._get_absolute_number(species, nic)
        if len(value) == 3:
            if np.linalg.det(value) == 0:
                warnings.warn('Error: Found linearly dependent constraints attached to atoms %s' % absolute_nr)
                continue
            fixed_atoms.append(absolute_nr)
        elif len(value) == 2:
            direction = np.cross(value[0], value[1])
            if np.linalg.norm(direction) == 0:
                warnings.warn('Error: Found linearly dependent constraints attached to atoms %s' % absolute_nr)
                continue
            constraint = ase.constraints.FixedLine(a=absolute_nr, direction=direction)
            constraints.append(constraint)
        elif len(value) == 1:
            constraint = ase.constraints.FixedPlane(a=absolute_nr, direction=np.array(value[0], dtype=np.float32))
            constraints.append(constraint)
        else:
            warnings.warn('Error: Found %s statements attached to atoms %s' % (len(value), absolute_nr))
    if fixed_atoms:
        constraints.append(ase.constraints.FixAtoms(indices=sorted(fixed_atoms)))
    if constraints:
        atoms.set_constraint(constraints)
    atoms.calc.atoms = atoms
    atoms.calc.push_oldstate()
    return atoms