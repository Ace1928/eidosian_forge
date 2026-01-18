import unittest
import numpy as np
import scipy.linalg
from skimage import data, img_as_float
from pygsp import graphs
def test_imgpatches(self):
    graphs.ImgPatches(img=self._img, patch_shape=(3, 3))