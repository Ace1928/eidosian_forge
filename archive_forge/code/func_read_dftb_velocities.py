import numpy as np
from ase.atoms import Atoms
from ase.utils import reader, writer
def read_dftb_velocities(atoms, filename):
    """Method to read velocities (AA/ps) from DFTB+ output file geo_end.xyz
    """
    from ase.units import second
    AngdivPs2ASE = 1.0 / (1e-12 * second)
    with open(filename) as fd:
        lines = fd.readlines()
    lines_ok = []
    for line in lines:
        if line.rstrip():
            lines_ok.append(line)
    velocities = []
    natoms = len(atoms)
    last_lines = lines_ok[-natoms:]
    for iline, line in enumerate(last_lines):
        inp = line.split()
        velocities.append([float(inp[5]) * AngdivPs2ASE, float(inp[6]) * AngdivPs2ASE, float(inp[7]) * AngdivPs2ASE])
    atoms.set_velocities(velocities)
    return atoms