from collections import OrderedDict
import numpy as np
import pytest
from typing import List, Tuple, Any
from ase.spectrum.dosdata import DOSData, GridDOSData, RawDOSData
@pytest.mark.parametrize('info, expected', sample_info)
def test_dosdata_init_info(self, info, expected):
    """Check 'info' parameter is handled properly"""
    if isinstance(expected, type) and isinstance(expected(), Exception):
        with pytest.raises(expected):
            dos_data = MinimalDOSData(info=info)
    else:
        dos_data = MinimalDOSData(info=info)
        assert dos_data.info == expected