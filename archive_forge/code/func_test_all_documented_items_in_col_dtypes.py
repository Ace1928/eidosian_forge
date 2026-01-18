import math
import re
import numpy as np
import pytest
import scipy.ndimage as ndi
from numpy.testing import (
from skimage import data, draw, transform
from skimage._shared import testing
from skimage.measure._regionprops import (
from skimage.segmentation import slic
def test_all_documented_items_in_col_dtypes():
    numpydoc_docscrape = pytest.importorskip('numpydoc.docscrape')
    docstring = numpydoc_docscrape.FunctionDoc(regionprops)
    notes_lines = docstring['Notes']
    property_lines = filter(lambda line: line.startswith('**'), notes_lines)
    pattern = '\\*\\*(?P<property_name>[a-z_]+)\\*\\*.*'
    property_names = {re.search(pattern, property_line).group('property_name') for property_line in property_lines}
    column_keys = set(COL_DTYPES.keys())
    assert column_keys == property_names