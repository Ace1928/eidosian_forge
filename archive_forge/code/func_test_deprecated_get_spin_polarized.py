from pathlib import Path
import pytest
from ase.calculators.calculator import Calculator
def test_deprecated_get_spin_polarized():
    calc = Calculator()
    with pytest.warns(FutureWarning):
        spinpol = calc.get_spin_polarized()
    assert spinpol is False