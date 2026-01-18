import unittest
import numpy as np
import scipy.linalg
from skimage import data, img_as_float
from pygsp import graphs
def test_grid2dimgpatches(self):
    graphs.Grid2dImgPatches(img=self._img, patch_shape=(3, 3))