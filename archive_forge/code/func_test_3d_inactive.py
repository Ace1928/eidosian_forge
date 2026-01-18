import numpy as np
from skimage._shared import testing
from skimage._shared._warnings import expected_warnings
from skimage._shared.testing import xfail, arch32
from skimage.segmentation import random_walker
from skimage.transform import resize
def test_3d_inactive():
    n = 30
    lx, ly, lz = (n, n, n)
    data, labels = make_3d_syntheticdata(lx, ly, lz)
    labels[5:25, 26:29, 26:29] = -1
    with expected_warnings(['"cg" mode|CObject type|scipy.sparse.linalg.cg']):
        labels = random_walker(data, labels, mode='cg')
    assert (labels.reshape(data.shape)[13:17, 13:17, 13:17] == 2).all()
    assert data.shape == labels.shape