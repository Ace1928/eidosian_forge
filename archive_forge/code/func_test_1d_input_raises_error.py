import inspect
import numpy as np
import pytest
from skimage import data, morphology, util
from skimage._shared._warnings import expected_warnings
from skimage._shared.testing import (
from skimage.filters import rank
from skimage.filters.rank import __all__ as all_rank_filters
from skimage.filters.rank import __3Dfilters as _3d_rank_filters
from skimage.filters.rank import subtract_mean
from skimage.morphology import ball, disk, gray
from skimage.util import img_as_float, img_as_ubyte
@pytest.mark.parametrize('func', [rank.autolevel, rank.equalize, rank.gradient, rank.maximum, rank.mean, rank.geometric_mean, rank.subtract_mean, rank.median, rank.minimum, rank.modal, rank.enhance_contrast, rank.pop, rank.sum, rank.threshold, rank.noise_filter, rank.entropy, rank.otsu, rank.majority])
def test_1d_input_raises_error(func):
    image = np.arange(10)
    footprint = disk(3)
    with pytest.raises(ValueError, match='`image` must have 2 or 3 dimensions, got 1'):
        func(image, footprint)