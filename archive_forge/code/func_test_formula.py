import numpy as np
import pytest
from ase import Atoms
from ase.formula import Formula
def test_formula():
    for sym in ['', 'Pu', 'Pu2', 'U2Pu2', 'U2((Pu2)2H)']:
        for mode in ['all', 'reduce', 'hill', 'metal']:
            for empirical in [False, True]:
                if empirical and mode in ['all', 'reduce']:
                    continue
                atoms = Atoms(sym)
                formula = atoms.get_chemical_formula(mode=mode, empirical=empirical)
                atoms2 = Atoms(formula)
                print(repr(sym), '->', repr(formula))
                n1 = np.sort(atoms.numbers)
                n2 = np.sort(atoms2.numbers)
                if empirical and len(atoms) > 0:
                    reduction = len(n1) // len(n2)
                    n2 = np.repeat(n2, reduction)
                assert (n1 == n2).all()