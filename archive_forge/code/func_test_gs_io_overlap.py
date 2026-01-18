import numpy as np
import pytest
from ase.vibrations import Vibrations
from ase.calculators.h2morse import (H2Morse, H2MorseCalculator,
from ase.calculators.h2morse import (H2MorseExcitedStatesCalculator,
def test_gs_io_overlap(testdir):
    """Test ground state IO and 'wave function' overlap"""
    atoms0 = H2Morse()
    calc0 = atoms0.calc
    fname = 'calc0'
    calc0.write(fname)
    calc1 = H2MorseCalculator(fname)
    for wf0, wf1 in zip(calc0.wfs, calc1.wfs):
        assert wf0 == pytest.approx(wf1, 1e-05)
    atoms1 = H2Morse()
    ov = calc0.overlap(calc0)
    assert np.eye(4) == pytest.approx(calc0.overlap(calc0), 1e-08)
    ov = calc0.overlap(atoms1.calc)
    assert np.eye(4) == pytest.approx(ov.dot(ov.T), 1e-08)