from collections import OrderedDict
import numpy as np
import pytest
from typing import List, Tuple, Any
from ase.spectrum.dosdata import DOSData, GridDOSData, RawDOSData
@pytest.mark.parametrize('info, expected', [({}, ''), ({'key1': 'value1'}, 'key1: value1'), (OrderedDict([('key1', 'value1'), ('key2', 'value2')]), 'key1: value1; key2: value2'), ({'key1': 'value1', 'label': 'xyz'}, 'xyz'), ({'label': 'xyz'}, 'xyz')])
def test_label_from_info(self, info, expected):
    assert DOSData.label_from_info(info) == expected