import numpy as np
from numpy.testing import assert_array_equal
import os
import os.path as osp
import shutil
import tempfile
import h5py as h5
from ..common import ut
from ..._hl.vds import vds_support
def generate_fem_stripe_image(self, value, dtype='uint16'):
    dset = np.empty(shape=self.fem_stripe_dimensions, dtype=dtype)
    dset.fill(value)
    return dset