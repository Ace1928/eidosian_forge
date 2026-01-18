import sys
import numpy as np
from math import factorial
from pytest import approx, fixture
from ase.build import molecule
from ase.calculators.emt import EMT
from ase.optimize import BFGS
from ase.vibrations import Vibrations
from ase.vibrations.franck_condon import (FranckCondonOverlap,
def test_ch4_minfreq(forces_a, relaxed, vibname):
    fc = FranckCondon(relaxed, vibname, minfreq=2000)
    nrel = 4
    FC, freq = fc.get_Franck_Condon_factors(293, forces_a)
    assert len(FC[0]) == 2 * nrel + 1
    assert len(freq[0]) == 2 * nrel + 1
    FC, freq = fc.get_Franck_Condon_factors(293, forces_a, 2)
    assert len(FC[1]) == 2 * nrel
    for i in range(3):
        assert len(freq[i]) == len(FC[i])