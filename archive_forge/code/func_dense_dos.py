from collections import OrderedDict
import numpy as np
import pytest
from typing import List, Tuple, Any
from ase.spectrum.dosdata import DOSData, GridDOSData, RawDOSData
@pytest.fixture
def dense_dos(self):
    x = np.linspace(0.0, 10.0, 11)
    y = np.sin(x / 10)
    return GridDOSData(x, y, info={'symbol': 'C', 'orbital': '2s', 'day': 'Tue'})