import numpy as np
import shutil
from ..util.dtype import img_as_bool
from ._registry import registry, registry_urls
from .. import __version__
import os.path as osp
import os
def nickel_solidification():
    """Image sequence of synchrotron x-radiographs showing the rapid
    solidification of a nickel alloy sample.

    Returns
    -------
    nickel_solidification: (11, 384, 512) uint16 ndarray

    Notes
    -----
    See info under `nickel_solidification.tif` at
    https://gitlab.com/scikit-image/data/-/blob/master/README.md#data.

    """
    return _load('data/solidification.tif')