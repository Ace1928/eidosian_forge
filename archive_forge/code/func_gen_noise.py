import os
import os.path as op
import nibabel as nb
import numpy as np
from math import floor, ceil
import itertools
import warnings
from .. import logging
from . import metrics as nam
from ..interfaces.base import (
from ..utils.filemanip import fname_presuffix, split_filename, ensure_list
from . import confounds
def gen_noise(self, image, mask=None, snr_db=10.0, dist='normal', bg_dist='normal'):
    """
        Generates a copy of an image with a certain amount of
        added gaussian noise (rayleigh for background in mask)
        """
    from math import sqrt
    snr = sqrt(np.power(10.0, snr_db / 10.0))
    if mask is None:
        mask = np.ones_like(image)
    else:
        mask[mask > 0] = 1
        mask[mask < 1] = 0
        if mask.ndim < image.ndim:
            mask = np.rollaxis(np.array([mask] * image.shape[3]), 0, 4)
    signal = image[mask > 0].reshape(-1)
    if dist == 'normal':
        signal = signal - signal.mean()
        sigma_n = sqrt(signal.var() / snr)
        noise = np.random.normal(size=image.shape, scale=sigma_n)
        if np.any(mask == 0) and bg_dist == 'rayleigh':
            bg_noise = np.random.rayleigh(size=image.shape, scale=sigma_n)
            noise[mask == 0] = bg_noise[mask == 0]
        im_noise = image + noise
    elif dist == 'rician':
        sigma_n = signal.mean() / snr
        n_1 = np.random.normal(size=image.shape, scale=sigma_n)
        n_2 = np.random.normal(size=image.shape, scale=sigma_n)
        stde_1 = n_1 / sqrt(2.0)
        stde_2 = n_2 / sqrt(2.0)
        im_noise = np.sqrt((image + stde_1) ** 2 + stde_2 ** 2)
    else:
        raise NotImplementedError('Only normal and rician distributions are supported')
    return im_noise