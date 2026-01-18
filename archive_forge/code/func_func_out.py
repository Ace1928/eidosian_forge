import numpy as np
import functools
from scipy import ndimage as ndi
from .._shared.utils import warn
@functools.wraps(func)
def func_out(image, footprint=None, *args, **kwargs):
    if footprint is None:
        footprint = ndi.generate_binary_structure(image.ndim, 1)
    return func(image, *args, footprint=footprint, **kwargs)