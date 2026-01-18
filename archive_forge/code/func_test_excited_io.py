import numpy as np
import pytest
from ase.vibrations import Vibrations
from ase.calculators.h2morse import (H2Morse, H2MorseCalculator,
from ase.calculators.h2morse import (H2MorseExcitedStatesCalculator,
def test_excited_io(testdir):
    """Check writing and reading"""
    fname = 'exlist.dat'
    atoms = H2Morse()
    exc = H2MorseExcitedStatesCalculator()
    exl1 = exc.calculate(atoms)
    exl1.write(fname)
    exl2 = H2MorseExcitedStates(fname)
    for ex1, ex2 in zip(exl1, exl2):
        assert ex1.energy == pytest.approx(ex2.energy, 0.001)
        assert ex1.mur == pytest.approx(ex2.mur, 1e-05)
        assert ex1.muv == pytest.approx(ex2.muv, 1e-05)