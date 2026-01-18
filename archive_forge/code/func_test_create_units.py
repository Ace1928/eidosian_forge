import numpy as np
import pytest
import ase.units
from ase.units import CODATA, create_units
import scipy.constants.codata
def test_create_units():
    """Check that units are created and allow attribute access."""
    new_units = ase.units.create_units(ase.units.__codata_version__)
    assert new_units.eV == new_units['eV'] == ase.units.eV
    for unit_name in new_units:
        assert getattr(new_units, unit_name) == getattr(ase.units, unit_name)
        assert new_units[unit_name] == getattr(ase.units, unit_name)