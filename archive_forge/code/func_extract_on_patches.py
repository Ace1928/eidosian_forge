from ..utils import *
import numpy as np
import scipy.misc
import scipy.io
from os.path import dirname
from os.path import join
def extract_on_patches(img, patch_size):
    h, w = img.shape
    patch_size = np.int(patch_size)
    patches = []
    for j in range(0, h - patch_size + 1, patch_size):
        for i in range(0, w - patch_size + 1, patch_size):
            patch = img[j:j + patch_size, i:i + patch_size]
            patches.append(patch)
    patches = np.array(patches)
    patch_features = []
    for p in patches:
        patch_features.append(_niqe_extract_subband_feats(p))
    patch_features = np.array(patch_features)
    return patch_features