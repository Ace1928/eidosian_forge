import pytest
from typing import Iterable
import numpy as np
from ase.spectrum.doscollection import (DOSCollection,
from ase.spectrum.dosdata import DOSData, RawDOSData, GridDOSData
@pytest.fixture
def another_rawdos(self):
    return RawDOSData([3.0, 2.0, 5.0], [1.0, 0.0, 2.0], info={'other_key': 'other_value'})