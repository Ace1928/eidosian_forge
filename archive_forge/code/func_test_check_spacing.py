from collections import OrderedDict
import numpy as np
import pytest
from typing import List, Tuple, Any
from ase.spectrum.dosdata import DOSData, GridDOSData, RawDOSData
def test_check_spacing(self, dense_dos):
    """Check a warning is logged when width < 2 * grid spacing"""
    dense_dos._sample([1], width=2.1)
    with pytest.warns(UserWarning, match='The broadening width is small'):
        dense_dos._sample([1], width=1.9)