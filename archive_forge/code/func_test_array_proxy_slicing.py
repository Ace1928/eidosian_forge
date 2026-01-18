from os.path import join as pjoin
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from .. import Nifti1Image, brikhead, load
from ..testing import assert_data_similar, data_path
from .test_fileslice import slicer_samples
def test_array_proxy_slicing(self):
    for tp in self.test_files:
        img = self.module.load(tp['fname'])
        arr = img.get_fdata()
        prox = img.dataobj
        assert prox.is_proxy
        for sliceobj in slicer_samples(img.shape):
            assert_array_equal(arr[sliceobj], prox[sliceobj])