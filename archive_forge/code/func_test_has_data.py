from copy import copy, deepcopy
import numpy as np
import pytest
import xarray as xr
import xarray.datatree_.datatree.testing as dtt
import xarray.testing as xrt
from xarray.core.datatree import DataTree
from xarray.core.treenode import NotFoundInTreeError
from xarray.tests import create_test_data, source_ndarray
def test_has_data(self):
    john: DataTree = DataTree(name='john', data=xr.Dataset({'a': 0}))
    assert john.has_data
    john_no_data: DataTree = DataTree(name='john', data=None)
    assert not john_no_data.has_data