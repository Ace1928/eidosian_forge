from ..utils import *
import numpy as np
import scipy.misc
import scipy.io
from os.path import dirname
from os.path import join
def get_patches_test_features(img, patch_size, stride=8):
    return _get_patches_generic(img, patch_size, 0, stride)