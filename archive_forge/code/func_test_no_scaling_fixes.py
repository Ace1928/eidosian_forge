import itertools
import logging
import os
import pickle
import re
from io import BytesIO, StringIO
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from .. import imageglobals
from ..analyze import AnalyzeHeader, AnalyzeImage
from ..arraywriters import WriterError
from ..casting import sctypes_aliases
from ..nifti1 import Nifti1Header
from ..optpkg import optional_package
from ..spatialimages import HeaderDataError, HeaderTypeError, supported_np_types
from ..testing import (
from ..tmpdirs import InTemporaryDirectory
from . import test_spatialimages as tsi
from . import test_wrapstruct as tws
def test_no_scaling_fixes(self):
    HC = self.header_class
    if not HC.has_data_slope:
        return
    hdr = HC()
    has_inter = HC.has_data_intercept
    slopes = (1, 0, np.nan, np.inf, -np.inf)
    inters = (0, np.nan, np.inf, -np.inf) if has_inter else (0,)
    for slope, inter in itertools.product(slopes, inters):
        hdr['scl_slope'] = slope
        if has_inter:
            hdr['scl_inter'] = inter
        self.assert_no_log_err(hdr)