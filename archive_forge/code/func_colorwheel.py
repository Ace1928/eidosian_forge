import numpy as np
import shutil
from ..util.dtype import img_as_bool
from ._registry import registry, registry_urls
from .. import __version__
import os.path as osp
import os
def colorwheel():
    """Color Wheel.

    Returns
    -------
    colorwheel : (370, 371, 3) uint8 image
        A colorwheel.
    """
    return _load('data/color.png')