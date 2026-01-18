import pytest
from typing import Iterable
import numpy as np
from ase.spectrum.doscollection import (DOSCollection,
from ase.spectrum.dosdata import DOSData, RawDOSData, GridDOSData
@pytest.fixture
def griddoscollection(self, griddos, another_griddos):
    return GridDOSCollection([griddos, another_griddos])