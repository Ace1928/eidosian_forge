import numpy as np
from numpy.testing import assert_array_equal
import os
import os.path as osp
import shutil
import tempfile
import h5py as h5
from ..common import ut
from ..._hl.vds import vds_support
@property
def sensor_module_dimensions(self):
    x_pixels = self.FEM_PIXELS_PER_CHIP_X * self.FEM_CHIPS_PER_STRIPE_X
    y_pixels = self.FEM_PIXELS_PER_CHIP_Y * self.FEM_CHIPS_PER_STRIPE_Y * self.FEM_STRIPES_PER_MODULE
    return (y_pixels, x_pixels)