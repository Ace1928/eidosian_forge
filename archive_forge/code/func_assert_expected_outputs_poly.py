from ...testing import utils
from ..confounds import TSNR
from .. import misc
import pytest
import numpy.testing as npt
from unittest import mock
import nibabel as nb
import numpy as np
import os
def assert_expected_outputs_poly(self, tsnrresult, expected_ranges):
    assert os.path.basename(tsnrresult.outputs.detrended_file) == self.out_filenames['detrended_file']
    self.assert_expected_outputs(tsnrresult, expected_ranges)