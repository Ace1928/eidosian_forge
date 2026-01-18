import numpy as np
import pytest
from numpy import sqrt, ceil
from numpy.testing import assert_almost_equal
from skimage import data
from skimage import img_as_float
from skimage.feature import daisy
def test_descs_shape():
    img = img_as_float(data.astronaut()[:256, :256].mean(axis=2))
    radius = 20
    step = 8
    descs = daisy(img, radius=radius, step=step)
    assert descs.shape[0] == ceil((img.shape[0] - radius * 2) / float(step))
    assert descs.shape[1] == ceil((img.shape[1] - radius * 2) / float(step))
    img = img[:-1, :-2]
    radius = 5
    step = 3
    descs = daisy(img, radius=radius, step=step)
    assert descs.shape[0] == ceil((img.shape[0] - radius * 2) / float(step))
    assert descs.shape[1] == ceil((img.shape[1] - radius * 2) / float(step))