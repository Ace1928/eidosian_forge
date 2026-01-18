import os
import struct
import unittest
import warnings
from io import BytesIO
import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal, assert_array_equal
from nibabel import nifti1 as nifti1
from nibabel.affines import from_matvec
from nibabel.casting import have_binary128, type_info
from nibabel.eulerangles import euler2mat
from nibabel.nifti1 import (
from nibabel.optpkg import optional_package
from nibabel.pkg_info import cmp_pkg_version
from nibabel.spatialimages import HeaderDataError
from nibabel.tmpdirs import InTemporaryDirectory
from ..freesurfer import load as mghload
from ..orientations import aff2axcodes
from ..testing import (
from . import test_analyze as tana
from . import test_spm99analyze as tspm
from .nibabel_data import get_nibabel_data, needs_nibabel_data
from .test_arraywriters import IUINT_TYPES, rt_err_estimate
from .test_orientations import ALL_ORNTS
def test_dynamic_dtype_aliases(self):
    for in_dt, mn, mx, alias, effective_dt in [(np.uint8, 0, 255, 'compat', np.uint8), (np.int8, 0, 127, 'compat', np.uint8), (np.int8, -128, 127, 'compat', np.int16), (np.int16, -32768, 32767, 'compat', np.int16), (np.uint16, 0, 32767, 'compat', np.int16), (np.uint16, 0, 65535, 'compat', np.int32), (np.int32, -2 ** 31, 2 ** 31 - 1, 'compat', np.int32), (np.uint32, 0, 2 ** 31 - 1, 'compat', np.int32), (np.uint32, 0, 2 ** 32 - 1, 'compat', None), (np.int64, -2 ** 31, 2 ** 31 - 1, 'compat', np.int32), (np.uint64, 0, 2 ** 31 - 1, 'compat', np.int32), (np.int64, 0, 2 ** 32 - 1, 'compat', None), (np.uint64, 0, 2 ** 32 - 1, 'compat', None), (np.float32, 0, 1e+30, 'compat', np.float32), (np.float64, 0, 1e+30, 'compat', np.float32), (np.float64, 0, 1e+40, 'compat', None), (np.int64, 0, 255, 'smallest', np.uint8), (np.int64, 0, 256, 'smallest', np.int16), (np.int64, -1, 255, 'smallest', np.int16), (np.int64, 0, 32768, 'smallest', np.int32), (np.int64, 0, 4294967296, 'smallest', None), (np.float32, 0, 1, 'smallest', None), (np.float64, 0, 1, 'smallest', None)]:
        arr = np.arange(24, dtype=in_dt).reshape((2, 3, 4))
        arr[0, 0, :2] = [mn, mx]
        img = self.image_class(arr, np.eye(4), dtype=alias)
        assert img.get_data_dtype() == alias
        if effective_dt is None:
            with pytest.raises(ValueError):
                img.get_data_dtype(finalize=True)
            continue
        assert img.get_data_dtype(finalize=True) == effective_dt
        assert img.get_data_dtype() == effective_dt
        img.set_data_dtype(alias)
        assert img.get_data_dtype() == alias
        img_rt = bytesio_round_trip(img)
        assert img_rt.get_data_dtype() == effective_dt
        assert img.get_data_dtype() == alias