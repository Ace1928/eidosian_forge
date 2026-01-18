from ase.atoms import Atoms
import numpy as np
from ase.data import atomic_numbers
from ase import units
from ase.utils import reader, writer
@reader
def read_gromacs(fd):
    """ From:
    http://manual.gromacs.org/current/online/gro.html
    C format
    "%5d%-5s%5s%5d%8.3f%8.3f%8.3f%8.4f%8.4f%8.4f"
    python: starting from 0, including first excluding last
    0:4 5:10 10:15 15:20 20:28 28:36 36:44 44:52 52:60 60:68

    Import gromacs geometry type files (.gro).
    Reads atom positions,
    velocities(if present) and
    simulation cell (if present)
    """
    atoms = Atoms()
    lines = fd.readlines()
    positions = []
    gromacs_velocities = []
    symbols = []
    tags = []
    gromacs_residuenumbers = []
    gromacs_residuenames = []
    gromacs_atomtypes = []
    sym2tag = {}
    tag = 0
    for line in lines[2:-1]:
        floatvect = (float(line[20:28]) * 10.0, float(line[28:36]) * 10.0, float(line[36:44]) * 10.0)
        positions.append(floatvect)
        velocities = np.array([0.0, 0.0, 0.0])
        vx = line[44:52].strip()
        vy = line[52:60].strip()
        vz = line[60:68].strip()
        for iv, vxyz in enumerate([vx, vy, vz]):
            if len(vxyz) > 0:
                try:
                    velocities[iv] = float(vxyz)
                except ValueError:
                    raise ValueError('can not convert velocity to float')
            else:
                velocities = None
        if velocities is not None:
            velocities *= units.nm / (1000.0 * units.fs)
            gromacs_velocities.append(velocities)
        gromacs_residuenumbers.append(int(line[0:5]))
        gromacs_residuenames.append(line[5:11].strip())
        symbol_read = line[11:16].strip()[0:2]
        if symbol_read not in sym2tag.keys():
            sym2tag[symbol_read] = tag
            tag += 1
        tags.append(sym2tag[symbol_read])
        if symbol_read in atomic_numbers:
            symbols.append(symbol_read)
        elif symbol_read[0] in atomic_numbers:
            symbols.append(symbol_read[0])
        elif symbol_read[-1] in atomic_numbers:
            symbols.append(symbol_read[-1])
        else:
            symbols.append('X')
        gromacs_atomtypes.append(line[11:16].strip())
    line = lines[-1]
    atoms = Atoms(symbols, positions, tags=tags)
    if len(gromacs_velocities) == len(atoms):
        atoms.set_velocities(gromacs_velocities)
    elif len(gromacs_velocities) != 0:
        raise ValueError('Some atoms velocities were not specified!')
    if not atoms.has('residuenumbers'):
        atoms.new_array('residuenumbers', gromacs_residuenumbers, int)
        atoms.set_array('residuenumbers', gromacs_residuenumbers, int)
    if not atoms.has('residuenames'):
        atoms.new_array('residuenames', gromacs_residuenames, str)
        atoms.set_array('residuenames', gromacs_residuenames, str)
    if not atoms.has('atomtypes'):
        atoms.new_array('atomtypes', gromacs_atomtypes, str)
        atoms.set_array('atomtypes', gromacs_atomtypes, str)
    atoms.pbc = False
    inp = lines[-1].split()
    try:
        grocell = list(map(float, inp))
    except ValueError:
        return atoms
    if len(grocell) < 3:
        return atoms
    cell = np.diag(grocell[:3])
    if len(grocell) >= 9:
        cell.flat[[1, 2, 3, 5, 6, 7]] = grocell[3:9]
    atoms.cell = cell * 10.0
    atoms.pbc = True
    return atoms