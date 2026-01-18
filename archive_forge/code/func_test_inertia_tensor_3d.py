import itertools
import numpy as np
import pytest
from scipy import ndimage as ndi
from skimage import draw
from skimage._shared import testing
from skimage._shared.testing import assert_allclose, assert_almost_equal, assert_equal
from skimage._shared.utils import _supported_float_type
from skimage.measure import (
def test_inertia_tensor_3d():
    image = draw.ellipsoid(10, 5, 3)
    T0 = inertia_tensor(image)
    eig0, V0 = np.linalg.eig(T0)
    v0 = V0[:, np.argmin(eig0)]
    assert np.allclose(v0, [1, 0, 0]) or np.allclose(-v0, [1, 0, 0])
    imrot = ndi.rotate(image.astype(float), 30, axes=(0, 1), order=1)
    Tr = inertia_tensor(imrot)
    eigr, Vr = np.linalg.eig(Tr)
    vr = Vr[:, np.argmin(eigr)]
    pi, cos, sin = (np.pi, np.cos, np.sin)
    R = np.array([[cos(pi / 6), -sin(pi / 6), 0], [sin(pi / 6), cos(pi / 6), 0], [0, 0, 1]])
    expected_vr = R @ v0
    assert np.allclose(vr, expected_vr, atol=0.001, rtol=0.01) or np.allclose(-vr, expected_vr, atol=0.001, rtol=0.01)