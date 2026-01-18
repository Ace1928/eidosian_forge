import numpy as np
import pytest
from ase.vibrations import Vibrations
from ase.calculators.h2morse import (H2Morse, H2MorseCalculator,
from ase.calculators.h2morse import (H2MorseExcitedStatesCalculator,
def test_excited_state():
    """Test excited state transition energies"""
    gsatoms = H2Morse()
    Egs0 = gsatoms.get_potential_energy()
    for i in range(1, 4):
        exatoms = H2Morse()
        exatoms[1].position[2] = Re[i]
        Egs = exatoms.get_potential_energy()
        exc = H2MorseExcitedStatesCalculator()
        exl = exc.calculate(exatoms)
        assert exl[i - 1].energy == pytest.approx(Etrans[i] - Egs + Egs0, 1e-08)