from __future__ import division
import logging
import warnings
import math
from base64 import b64encode
import numpy as np
import PIL.Image
import ipywidgets
import ipywebrtc
from ipython_genutils.py3compat import string_types
from ipyvolume import utils
def color_to_binary_or_json(ar, obj=None):
    if ar is None:
        return None
    element = ar
    dimension = 0
    try:
        while True:
            element = element[0]
            dimension += 1
    except:
        pass
    try:
        element = element.item()
    except:
        pass
    if isinstance(element, string_types):
        return array_to_json(ar)
    if dimension == 0:
        return ar
    if ar.ndim > 1 and ar.shape[-1] == 3:
        ones = np.ones(ar.shape[:-1])
        ar = np.stack([ar[..., 0], ar[..., 1], ar[..., 2], ones], axis=-1)
    elif ar.ndim > 1 and ar.shape[-1] != 4:
        raise ValueError('array should be of shape (...,3) or (...,4), not %r' % (ar.shape,))
    if dimension == 3:
        return [array_to_binary(ar[k]) for k in range(len(ar))]
    else:
        return [array_to_binary(ar)]