import numpy as np
import functools
from scipy import ndimage as ndi
from .._shared.utils import warn
def default_footprint(func):
    """Decorator to add a default footprint to morphology functions.

    Parameters
    ----------
    func : function
        A morphology function such as erosion, dilation, opening, closing,
        white_tophat, or black_tophat.

    Returns
    -------
    func_out : function
        The function, using a default footprint of same dimension
        as the input image with connectivity 1.

    """

    @functools.wraps(func)
    def func_out(image, footprint=None, *args, **kwargs):
        if footprint is None:
            footprint = ndi.generate_binary_structure(image.ndim, 1)
        return func(image, *args, footprint=footprint, **kwargs)
    return func_out