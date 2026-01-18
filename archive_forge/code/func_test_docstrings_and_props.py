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
def test_docstrings_and_props():

    def foo():
        """foo"""
    has_docstrings = bool(foo.__doc__)
    region = regionprops(SAMPLE)[0]
    docs = _parse_docs()
    props = [m for m in dir(region) if not m.startswith('_')]
    nr_docs_parsed = len(docs)
    nr_props = len(props)
    if has_docstrings:
        assert_equal(nr_docs_parsed, nr_props)
        ds = docs['moments_weighted_normalized']
        assert 'iteration' not in ds
        assert len(ds.split('\n')) > 3
    else:
        assert_equal(nr_docs_parsed, 0)