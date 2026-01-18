from ase.parallel import paropen
from ase.calculators.lammps.unitconvert import convert
def lammps_create_atoms(fileobj, parameters, atoms, prismobj):
    """Create atoms in lammps with 'create_box' and 'create_atoms'

    :param fileobj: open stream for lammps input
    :param parameters: dict of all lammps parameters
    :type parameters: dict
    :param atoms: Atoms object
    :type atoms: Atoms
    :param prismobj: coordinate transformation between ase and lammps
    :type prismobj: Prism

    """
    if parameters['verbose']:
        fileobj.write('## Original ase cell\n'.encode('utf-8'))
        fileobj.write(''.join(['# {0:.16} {1:.16} {2:.16}\n'.format(*x) for x in atoms.get_cell()]).encode('utf-8'))
    fileobj.write('lattice sc 1.0\n'.encode('utf-8'))
    xhi, yhi, zhi, xy, xz, yz = convert(prismobj.get_lammps_prism(), 'distance', 'ASE', parameters.units)
    if parameters['always_triclinic'] or prismobj.is_skewed():
        fileobj.write('region asecell prism 0.0 {0} 0.0 {1} 0.0 {2} '.format(xhi, yhi, zhi).encode('utf-8'))
        fileobj.write('{0} {1} {2} side in units box\n'.format(xy, xz, yz).encode('utf-8'))
    else:
        fileobj.write('region asecell block 0.0 {0} 0.0 {1} 0.0 {2} side in units box\n'.format(xhi, yhi, zhi).encode('utf-8'))
    symbols = atoms.get_chemical_symbols()
    try:
        species = parameters['specorder']
    except AttributeError:
        species = sorted(set(symbols))
    species_i = {s: i + 1 for i, s in enumerate(species)}
    fileobj.write('create_box {0} asecell\n'.format(len(species)).encode('utf-8'))
    for sym, pos in zip(symbols, atoms.get_positions()):
        pos = convert(pos, 'distance', 'ASE', parameters.units)
        if parameters['verbose']:
            fileobj.write('# atom pos in ase cell: {0:.16} {1:.16} {2:.16}\n'.format(*tuple(pos)).encode('utf-8'))
        fileobj.write('create_atoms {0} single {1} {2} {3} remap yes units box\n'.format(*(species_i[sym],) + tuple(prismobj.vector_to_lammps(pos))).encode('utf-8'))