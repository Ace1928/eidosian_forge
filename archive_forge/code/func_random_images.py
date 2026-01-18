from __future__ import annotations
import os
from contextlib import contextmanager
import pytest
import numpy as np
from skimage.io import imsave
from dask.array.image import imread as da_imread
from dask.utils import tmpdir
@contextmanager
def random_images(n, shape):
    with tmpdir() as dirname:
        for i in range(n):
            fn = os.path.join(dirname, 'image.%d.png' % i)
            x = np.random.randint(0, 255, size=shape).astype('u1')
            imsave(fn, x, check_contrast=False)
        yield os.path.join(dirname, '*.png')