import itertools
import unittest
from io import BytesIO
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from ..optpkg import optional_package
from ..casting import sctypes_aliases, shared_range, type_info
from ..spatialimages import HeaderDataError
from ..spm99analyze import HeaderTypeError, Spm99AnalyzeHeader, Spm99AnalyzeImage
from ..testing import (
from ..volumeutils import _dt_min_max, apply_read_scaling
from . import test_analyze
def assert_scaling_equal(self, hdr, slope, inter):
    h_slope, h_inter = self._get_raw_scaling(hdr)
    assert_array_equal(h_slope, slope)
    assert_array_equal(h_inter, inter)