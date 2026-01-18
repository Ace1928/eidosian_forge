from copy import copy, deepcopy
import numpy as np
import pytest
import xarray as xr
import xarray.datatree_.datatree.testing as dtt
import xarray.testing as xrt
from xarray.core.datatree import DataTree
from xarray.core.treenode import NotFoundInTreeError
from xarray.tests import create_test_data, source_ndarray
def test_copy_subtree(self):
    dt = DataTree.from_dict({'/level1/level2/level3': xr.Dataset()})
    actual = dt['/level1/level2'].copy()
    expected = DataTree.from_dict({'/level3': xr.Dataset()}, name='level2')
    dtt.assert_identical(actual, expected)