import numpy as np
from ase import Atoms
from ase.units import Bohr, Ry
from ase.utils import reader, writer
def read_scf(filename):
    try:
        with open(filename + '.scf', 'r') as fd:
            pip = fd.readlines()
        ene = []
        for line in pip:
            if line[0:4] == ':ENE':
                ene.append(float(line[43:59]) * Ry)
        return ene
    except Exception:
        return None