import numpy as np
from skimage._shared import testing
from skimage._shared._warnings import expected_warnings
from skimage._shared.testing import xfail, arch32
from skimage.segmentation import random_walker
from skimage.transform import resize
def make_3d_syntheticdata(lx, ly=None, lz=None):
    if ly is None:
        ly = lx
    if lz is None:
        lz = lx
    np.random.seed(1234)
    data = np.zeros((lx, ly, lz)) + 0.1 * np.random.randn(lx, ly, lz)
    small_l = int(lx // 5)
    data[lx // 2 - small_l:lx // 2 + small_l, ly // 2 - small_l:ly // 2 + small_l, lz // 2 - small_l:lz // 2 + small_l] = 1
    data[lx // 2 - small_l + 1:lx // 2 + small_l - 1, ly // 2 - small_l + 1:ly // 2 + small_l - 1, lz // 2 - small_l + 1:lz // 2 + small_l - 1] = 0
    hole_size = np.max([1, small_l // 8])
    data[lx // 2 - small_l, ly // 2 - hole_size:ly // 2 + hole_size, lz // 2 - hole_size:lz // 2 + hole_size] = 0
    seeds = np.zeros_like(data)
    seeds[lx // 5, ly // 5, lz // 5] = 1
    seeds[lx // 2 + small_l // 4, ly // 2 - small_l // 4, lz // 2 - small_l // 4] = 2
    return (data, seeds)