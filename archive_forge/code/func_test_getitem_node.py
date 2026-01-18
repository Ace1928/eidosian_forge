from copy import copy, deepcopy
import numpy as np
import pytest
import xarray as xr
import xarray.datatree_.datatree.testing as dtt
import xarray.testing as xrt
from xarray.core.datatree import DataTree
from xarray.core.treenode import NotFoundInTreeError
from xarray.tests import create_test_data, source_ndarray
def test_getitem_node(self):
    folder1: DataTree = DataTree(name='folder1')
    results: DataTree = DataTree(name='results', parent=folder1)
    highres: DataTree = DataTree(name='highres', parent=results)
    assert folder1['results'] is results
    assert folder1['results/highres'] is highres