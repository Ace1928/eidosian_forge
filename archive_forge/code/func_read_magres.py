import re
import numpy as np
from collections import OrderedDict
import ase.units
from ase.atoms import Atoms
from ase.spacegroup import Spacegroup
from ase.spacegroup.spacegroup import SpacegroupNotFoundError
from ase.calculators.singlepoint import SinglePointDFTCalculator
def read_magres(fd, include_unrecognised=False):
    """
        Reader function for magres files.
    """
    blocks_re = re.compile('[\\[<](?P<block_name>.*?)[>\\]](.*?)[<\\[]/' + '(?P=block_name)[\\]>]', re.M | re.S)
    '\n    Here are defined the various functions required to parse\n    different blocks.\n    '

    def tensor33(x):
        return np.squeeze(np.reshape(x, (3, 3))).tolist()

    def tensor31(x):
        return np.squeeze(np.reshape(x, (3, 1))).tolist()

    def get_version(file_contents):
        """
            Look for and parse the magres file format version line
        """
        lines = file_contents.split('\n')
        match = re.match('\\#\\$magres-abinitio-v([0-9]+).([0-9]+)', lines[0])
        if match:
            version = match.groups()
            version = tuple((vnum for vnum in version))
        else:
            version = None
        return version

    def parse_blocks(file_contents):
        """
            Parse series of XML-like deliminated blocks into a list of
            (block_name, contents) tuples
        """
        blocks = blocks_re.findall(file_contents)
        return blocks

    def parse_block(block):
        """
            Parse block contents into a series of (tag, data) records
        """

        def clean_line(line):
            line = re.sub('#(.*?)\n', '', line)
            line = line.strip()
            return line
        name, data = block
        lines = [clean_line(line) for line in data.split('\n')]
        records = []
        for line in lines:
            xs = line.split()
            if len(xs) > 0:
                tag = xs[0]
                data = xs[1:]
                records.append((tag, data))
        return (name, records)

    def check_units(d):
        """
            Verify that given units for a particular tag are correct.
        """
        allowed_units = {'lattice': 'Angstrom', 'atom': 'Angstrom', 'ms': 'ppm', 'efg': 'au', 'efg_local': 'au', 'efg_nonlocal': 'au', 'isc': '10^19.T^2.J^-1', 'isc_fc': '10^19.T^2.J^-1', 'isc_orbital_p': '10^19.T^2.J^-1', 'isc_orbital_d': '10^19.T^2.J^-1', 'isc_spin': '10^19.T^2.J^-1', 'isc': '10^19.T^2.J^-1', 'sus': '10^-6.cm^3.mol^-1', 'calc_cutoffenergy': 'Hartree'}
        if d[0] in d and d[1] == allowed_units[d[0]]:
            pass
        else:
            raise RuntimeError('Unrecognized units: %s %s' % (d[0], d[1]))
        return d

    def parse_magres_block(block):
        """
            Parse magres block into data dictionary given list of record
            tuples.
        """
        name, records = block

        def ntensor33(name):
            return lambda d: {name: tensor33([float(x) for x in data])}

        def sitensor33(name):
            return lambda d: {'atom': {'label': data[0], 'index': int(data[1])}, name: tensor33([float(x) for x in data[2:]])}

        def sisitensor33(name):
            return lambda d: {'atom1': {'label': data[0], 'index': int(data[1])}, 'atom2': {'label': data[2], 'index': int(data[3])}, name: tensor33([float(x) for x in data[4:]])}
        tags = {'ms': sitensor33('sigma'), 'sus': ntensor33('S'), 'efg': sitensor33('V'), 'efg_local': sitensor33('V'), 'efg_nonlocal': sitensor33('V'), 'isc': sisitensor33('K'), 'isc_fc': sisitensor33('K'), 'isc_spin': sisitensor33('K'), 'isc_orbital_p': sisitensor33('K'), 'isc_orbital_d': sisitensor33('K'), 'units': check_units}
        data_dict = {}
        for record in records:
            tag, data = record
            if tag not in data_dict:
                data_dict[tag] = []
            data_dict[tag].append(tags[tag](data))
        return data_dict

    def parse_atoms_block(block):
        """
            Parse atoms block into data dictionary given list of record tuples.
        """
        name, records = block

        def lattice(d):
            return tensor33([float(x) for x in data])

        def atom(d):
            return {'species': data[0], 'label': data[1], 'index': int(data[2]), 'position': tensor31([float(x) for x in data[3:]])}

        def symmetry(d):
            return ' '.join(data)
        tags = {'lattice': lattice, 'atom': atom, 'units': check_units, 'symmetry': symmetry}
        data_dict = {}
        for record in records:
            tag, data = record
            if tag not in data_dict:
                data_dict[tag] = []
            data_dict[tag].append(tags[tag](data))
        return data_dict

    def parse_generic_block(block):
        """
            Parse any other block into data dictionary given list of record
            tuples.
        """
        name, records = block
        data_dict = {}
        for record in records:
            tag, data = record
            if tag not in data_dict:
                data_dict[tag] = []
            data_dict[tag].append(data)
        return data_dict
    '\n        Actual parser code.\n    '
    block_parsers = {'magres': parse_magres_block, 'atoms': parse_atoms_block, 'calculation': parse_generic_block}
    file_contents = fd.read()
    version = get_version(file_contents)
    if version is None:
        raise RuntimeError('File is not in standard Magres format')
    blocks = parse_blocks(file_contents)
    data_dict = {}
    for block_data in blocks:
        block = parse_block(block_data)
        if block[0] in block_parsers:
            block_dict = block_parsers[block[0]](block)
            data_dict[block[0]] = block_dict
        elif include_unrecognised:
            data_dict[block[0]] = block_data[1]
    if 'atoms' not in data_dict:
        raise RuntimeError('Magres file does not contain structure data')
    magres_units = {'Angstrom': ase.units.Ang}
    if 'lattice' in data_dict['atoms']:
        try:
            u = dict(data_dict['atoms']['units'])['lattice']
        except KeyError:
            raise RuntimeError('No units detected in file for lattice')
        u = magres_units[u]
        cell = np.array(data_dict['atoms']['lattice'][0]) * u
        pbc = True
    else:
        cell = None
        pbc = False
    symbols = []
    positions = []
    indices = []
    labels = []
    if 'atom' in data_dict['atoms']:
        try:
            u = dict(data_dict['atoms']['units'])['atom']
        except KeyError:
            raise RuntimeError('No units detected in file for atom positions')
        u = magres_units[u]
        custom_species = None
        for a in data_dict['atoms']['atom']:
            spec_custom = a['species'].split(':', 1)
            if len(spec_custom) > 1 and custom_species is None:
                custom_species = list(symbols)
            symbols.append(spec_custom[0])
            positions.append(a['position'])
            indices.append(a['index'])
            labels.append(a['label'])
            if custom_species is not None:
                custom_species.append(a['species'])
    atoms = Atoms(cell=cell, pbc=pbc, symbols=symbols, positions=positions)
    if custom_species is not None:
        atoms.new_array('castep_custom_species', np.array(custom_species))
    if 'symmetry' in data_dict['atoms']:
        try:
            spg = Spacegroup(data_dict['atoms']['symmetry'][0])
        except SpacegroupNotFoundError:
            spg = Spacegroup(1)
        atoms.info['spacegroup'] = spg
    atoms.new_array('indices', np.array(indices))
    atoms.new_array('labels', np.array(labels))
    li_list = list(zip(labels, indices))

    def create_magres_array(name, order, block):
        if order == 1:
            u_arr = [None] * len(li_list)
        elif order == 2:
            u_arr = [[None] * (i + 1) for i in range(len(li_list))]
        else:
            raise ValueError('Invalid order value passed to create_magres_array')
        for s in block:
            if order == 1:
                at = (s['atom']['label'], s['atom']['index'])
                try:
                    ai = li_list.index(at)
                except ValueError:
                    raise RuntimeError('Invalid data in magres block')
                u_arr[ai] = s[mn]
            else:
                at1 = (s['atom1']['label'], s['atom1']['index'])
                at2 = (s['atom2']['label'], s['atom2']['index'])
                ai1 = li_list.index(at1)
                ai2 = li_list.index(at2)
                ai1, ai2 = sorted((ai1, ai2), reverse=True)
                u_arr[ai1][ai2] = s[mn]
        if order == 1:
            return np.array(u_arr)
        else:
            return np.array(u_arr, dtype=object)
    if 'magres' in data_dict:
        if 'units' in data_dict['magres']:
            atoms.info['magres_units'] = dict(data_dict['magres']['units'])
            for u in atoms.info['magres_units']:
                u0 = u.split('_')[0]
                if u0 not in _mprops:
                    raise RuntimeError('Invalid data in magres block')
                mn, order = _mprops[u0]
                if order > 0:
                    u_arr = create_magres_array(mn, order, data_dict['magres'][u])
                    atoms.new_array(u, u_arr)
                elif atoms.calc is None:
                    calc = SinglePointDFTCalculator(atoms)
                    atoms.calc = calc
                    atoms.calc.results[u] = data_dict['magres'][u][0][mn]
    if 'calculation' in data_dict:
        atoms.info['magresblock_calculation'] = data_dict['calculation']
    if include_unrecognised:
        for b in data_dict:
            if b not in block_parsers:
                atoms.info['magresblock_' + b] = data_dict[b]
    return atoms