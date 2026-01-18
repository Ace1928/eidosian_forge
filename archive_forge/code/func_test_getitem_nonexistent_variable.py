from copy import copy, deepcopy
import numpy as np
import pytest
import xarray as xr
import xarray.datatree_.datatree.testing as dtt
import xarray.testing as xrt
from xarray.core.datatree import DataTree
from xarray.core.treenode import NotFoundInTreeError
from xarray.tests import create_test_data, source_ndarray
def test_getitem_nonexistent_variable(self):
    data = xr.Dataset({'temp': [0, 50]})
    results: DataTree = DataTree(name='results', data=data)
    with pytest.raises(KeyError):
        results['pressure']