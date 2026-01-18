import numpy as np
import pytest
from scipy import ndimage
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.image import (
@pytest.fixture
def downsampled_face(raccoon_face_fxt):
    face = raccoon_face_fxt
    face = face[::2, ::2] + face[1::2, ::2] + face[::2, 1::2] + face[1::2, 1::2]
    face = face[::2, ::2] + face[1::2, ::2] + face[::2, 1::2] + face[1::2, 1::2]
    face = face.astype(np.float32)
    face /= 16.0
    return face