import sys
import numpy as np
from math import factorial
from pytest import approx, fixture
from ase.build import molecule
from ase.calculators.emt import EMT
from ase.optimize import BFGS
from ase.vibrations import Vibrations
from ase.vibrations.franck_condon import (FranckCondonOverlap,
def test_ch4_all(forces_a, relaxed, vibname):
    """Evaluate Franck-Condon overlaps in
    a molecule suddenly exposed to a different potential"""
    fc = FranckCondon(relaxed, vibname)
    ndof = 3 * len(relaxed)
    HR_a, f_a = fc.get_Huang_Rhys_factors(forces_a)
    assert len(HR_a) == ndof
    assert HR_a[:-1] == approx(0, abs=1e-10)
    assert HR_a[-1] == approx(0.859989171)
    FC, freq = fc.get_Franck_Condon_factors(293, forces_a)
    assert len(FC[0]) == 2 * ndof + 1
    assert len(freq[0]) == 2 * ndof + 1