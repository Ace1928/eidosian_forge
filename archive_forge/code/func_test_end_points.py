import numpy as np
import pytest
from numpy.testing import assert_equal, assert_allclose
from skimage import data
from skimage._shared.utils import _supported_float_type
from skimage.color import rgb2gray
from skimage.filters import gaussian
from skimage.segmentation import active_contour
def test_end_points():
    img = data.astronaut()
    img = rgb2gray(img)
    s = np.linspace(0, 2 * np.pi, 400)
    r = 100 + 100 * np.sin(s)
    c = 220 + 100 * np.cos(s)
    init = np.array([r, c]).T
    snake = active_contour(gaussian(img, sigma=3), init, boundary_condition='periodic', alpha=0.015, beta=10, w_line=0, w_edge=1, gamma=0.001, max_num_iter=100)
    assert np.sum(np.abs(snake[0, :] - snake[-1, :])) < 2
    snake = active_contour(gaussian(img, sigma=3), init, boundary_condition='free', alpha=0.015, beta=10, w_line=0, w_edge=1, gamma=0.001, max_num_iter=100)
    assert np.sum(np.abs(snake[0, :] - snake[-1, :])) > 2
    snake = active_contour(gaussian(img, sigma=3), init, boundary_condition='fixed', alpha=0.015, beta=10, w_line=0, w_edge=1, gamma=0.001, max_num_iter=100)
    assert_allclose(snake[0, :], [r[0], c[0]], atol=1e-05)