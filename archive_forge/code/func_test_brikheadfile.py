from os.path import join as pjoin
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from .. import Nifti1Image, brikhead, load
from ..testing import assert_data_similar, data_path
from .test_fileslice import slicer_samples
def test_brikheadfile(self):
    for tp in self.test_files:
        with pytest.raises(tp['err']):
            self.module.load(tp['head'])