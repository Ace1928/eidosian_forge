from collections import OrderedDict
import numpy as np
import pytest
from typing import List, Tuple, Any
from ase.spectrum.dosdata import DOSData, GridDOSData, RawDOSData
@pytest.fixture
def sparse_dos(self):
    return RawDOSData([1.2, 3.4, 5.0], [3.0, 2.1, 0.0], info={'symbol': 'H', 'number': '1', 'food': 'egg'})