from ...testing import utils
from ..confounds import TSNR
from .. import misc
import pytest
import numpy.testing as npt
from unittest import mock
import nibabel as nb
import numpy as np
import os
def assert_default_outputs(self, outputs):
    assert os.path.basename(outputs.mean_file) == self.out_filenames['mean_file']
    assert os.path.basename(outputs.stddev_file) == self.out_filenames['stddev_file']
    assert os.path.basename(outputs.tsnr_file) == self.out_filenames['tsnr_file']