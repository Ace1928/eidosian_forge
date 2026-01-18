from ...testing import utils
from ..confounds import TSNR
from .. import misc
import pytest
import numpy.testing as npt
from unittest import mock
import nibabel as nb
import numpy as np
import os
def assert_expected_outputs(self, tsnrresult, expected_ranges):
    self.assert_default_outputs(tsnrresult.outputs)
    self.assert_unchanged(expected_ranges)