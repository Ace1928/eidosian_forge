import pytest
from ase.vibrations.resonant_raman import ResonantRamanCalculator
from ase.calculators.h2morse import (H2Morse,
from ase.vibrations.albrecht import Albrecht
@pytest.fixture
def rrname(atoms):
    """Prepare the Resonant Raman calculation"""
    name = 'rrmorse'
    with ResonantRamanCalculator(atoms, H2MorseExcitedStatesCalculator, overlap=lambda x, y: x.overlap(y), name=name, txt='-') as rmc:
        rmc.run()
    return name