import numpy as np
import h5py
from .common import ut, TestCase
def test_initial_swmr_mode_off(self):
    """ Verify that the file is not initially in SWMR mode"""
    self.assertFalse(self.f.swmr_mode)