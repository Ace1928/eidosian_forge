import numpy as np
import shutil
from ..util.dtype import img_as_bool
from ._registry import registry, registry_urls
from .. import __version__
import os.path as osp
import os
def horse():
    """Black and white silhouette of a horse.

    This image was downloaded from
    `openclipart <http://openclipart.org/detail/158377/horse-by-marauder>`

    No copyright restrictions. CC0 given by owner (Andreas Preuss (marauder)).

    Returns
    -------
    horse : (328, 400) bool ndarray
        Horse image.
    """
    return img_as_bool(_load('data/horse.png', as_gray=True))