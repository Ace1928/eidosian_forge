import os
import warnings
from pathlib import Path
from unittest import TestCase
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from ..ecat import (
from ..openers import Opener
from ..testing import data_path, suppress_warnings
from ..tmpdirs import InTemporaryDirectory
from . import test_wrapstruct as tws
from .test_fileslice import slicer_samples
def test_data_regression(self):
    vals = dict(max=248750736458.0, min=1125342630.0, mean=117907565661.46666)
    data = self.img.get_fdata()
    assert data.max() == vals['max']
    assert data.min() == vals['min']
    assert_array_almost_equal(data.mean(), vals['mean'])