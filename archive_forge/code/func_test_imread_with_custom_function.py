from __future__ import annotations
import os
from contextlib import contextmanager
import pytest
import numpy as np
from skimage.io import imsave
from dask.array.image import imread as da_imread
from dask.utils import tmpdir
def test_imread_with_custom_function():

    def imread2(fn):
        return np.ones((2, 3, 4), dtype='i1')
    with random_images(4, (5, 6, 3)) as globstring:
        im = da_imread(globstring, imread=imread2)
        assert (im.compute() == np.ones((4, 2, 3, 4), dtype='u1')).all()