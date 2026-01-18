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
def array_sequence_to_binary_or_json(ar, obj=None):
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
        if isinstance(element, np.ndarray):
            return []
        else:
            return element
    if isinstance(ar, (list, tuple, np.ndarray)):
        if isinstance(ar[0], (list, tuple, np.ndarray)):
            return [array_to_binary(ar[k]) for k in range(len(ar))]
        else:
            return [array_to_binary(ar)]
    else:
        raise ValueError('Expected a sequence, got %r', ar)