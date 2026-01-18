import numpy as np
import pytest
from scipy import ndimage
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.image import (
def test_extract_patches_square(downsampled_face):
    face = downsampled_face
    i_h, i_w = face.shape
    p = 8
    expected_n_patches = (i_h - p + 1, i_w - p + 1)
    patches = _extract_patches(face, patch_shape=p)
    assert patches.shape == (expected_n_patches[0], expected_n_patches[1], p, p)