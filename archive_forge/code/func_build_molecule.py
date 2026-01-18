import sys
import numpy as np
from ase.db import connect
from ase.build import bulk
from ase.io import read, write
from ase.visualize import view
from ase.build import molecule
from ase.atoms import Atoms
from ase.symbols import string2symbols
from ase.data import ground_state_magnetic_moments
from ase.data import atomic_numbers, covalent_radii
def build_molecule(args):
    try:
        atoms = molecule(args.name)
    except (NotImplementedError, KeyError):
        symbols = string2symbols(args.name)
        if len(symbols) == 1:
            Z = atomic_numbers[symbols[0]]
            magmom = ground_state_magnetic_moments[Z]
            atoms = Atoms(args.name, magmoms=[magmom])
        elif len(symbols) == 2:
            if args.bond_length is None:
                b = covalent_radii[atomic_numbers[symbols[0]]] + covalent_radii[atomic_numbers[symbols[1]]]
            else:
                b = args.bond_length
            atoms = Atoms(args.name, positions=[(0, 0, 0), (b, 0, 0)])
        else:
            raise ValueError('Unknown molecule: ' + args.name)
    else:
        if len(atoms) == 2 and args.bond_length is not None:
            atoms.set_distance(0, 1, args.bond_length)
    if args.unit_cell is None:
        if args.vacuum:
            atoms.center(vacuum=args.vacuum)
        else:
            atoms.center(about=[0, 0, 0])
    else:
        a = [float(x) for x in args.unit_cell.split(',')]
        if len(a) == 1:
            cell = [a[0], a[0], a[0]]
        elif len(a) == 3:
            cell = a
        else:
            a, b, c, alpha, beta, gamma = a
            degree = np.pi / 180.0
            cosa = np.cos(alpha * degree)
            cosb = np.cos(beta * degree)
            sinb = np.sin(beta * degree)
            cosg = np.cos(gamma * degree)
            sing = np.sin(gamma * degree)
            cell = [[a, 0, 0], [b * cosg, b * sing, 0], [c * cosb, c * (cosa - cosb * cosg) / sing, c * np.sqrt(sinb ** 2 - ((cosa - cosb * cosg) / sing) ** 2)]]
        atoms.cell = cell
        atoms.center()
    atoms.pbc = args.periodic
    return atoms