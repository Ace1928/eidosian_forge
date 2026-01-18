import shutil
import unittest
from os.path import join as pjoin
from tempfile import mkdtemp
import numpy as np
from ..loadsave import load, save
from ..nifti1 import Nifti1Image
@unittest.skipIf(SOFT_LIMIT > 4900, 'It would take too long to test filehandles')
def test_multiload():
    N = SOFT_LIMIT + 100
    arr = np.arange(24, dtype='int32').reshape((2, 3, 4))
    img = Nifti1Image(arr, np.eye(4))
    imgs = []
    try:
        tmpdir = mkdtemp()
        fname = pjoin(tmpdir, 'test.img')
        save(img, fname)
        for i in range(N):
            imgs.append(load(fname))
    finally:
        del img, imgs
        shutil.rmtree(tmpdir)