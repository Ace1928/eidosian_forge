import numpy as np
import pytest
from scipy import ndimage
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.image import (
def test_extract_patch_same_size_image(downsampled_face):
    face = downsampled_face
    patches = extract_patches_2d(face, face.shape, max_patches=2)
    assert patches.shape[0] == 1