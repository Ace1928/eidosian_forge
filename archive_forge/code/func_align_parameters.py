import json
import logging
import random
import warnings
import numpy as np
from ..base import numeric_types
from .. import ndarray as nd
from ..ndarray._internal import _cvcopyMakeBorder as copyMakeBorder
from .. import io
from .image import RandomOrderAug, ColorJitterAug, LightingAug, ColorNormalizeAug
from .image import ResizeAug, ForceResizeAug, CastAug, HueJitterAug, RandomGrayAug
from .image import fixed_crop, ImageIter, Augmenter
from ..util import is_np_array
from .. import numpy as _mx_np  # pylint: disable=reimported
def align_parameters(params):
    """Align parameters as pairs"""
    out_params = []
    num = 1
    for p in params:
        if not isinstance(p, list):
            p = [p]
        out_params.append(p)
        num = max(num, len(p))
    for k, p in enumerate(out_params):
        if len(p) != num:
            assert len(p) == 1
            out_params[k] = p * num
    return out_params