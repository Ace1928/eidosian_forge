from copy import copy, deepcopy
import numpy as np
import pytest
import xarray as xr
import xarray.datatree_.datatree.testing as dtt
import xarray.testing as xrt
from xarray.core.datatree import DataTree
from xarray.core.treenode import NotFoundInTreeError
from xarray.tests import create_test_data, source_ndarray
def test_bad_names(self):
    with pytest.raises(TypeError):
        DataTree(name=5)
    with pytest.raises(ValueError):
        DataTree(name='folder/data')