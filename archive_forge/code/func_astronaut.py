import numpy as np
import shutil
from ..util.dtype import img_as_bool
from ._registry import registry, registry_urls
from .. import __version__
import os.path as osp
import os
def astronaut():
    """Color image of the astronaut Eileen Collins.

    Photograph of Eileen Collins, an American astronaut. She was selected
    as an astronaut in 1992 and first piloted the space shuttle STS-63 in
    1995. She retired in 2006 after spending a total of 38 days, 8 hours
    and 10 minutes in outer space.

    This image was downloaded from the NASA Great Images database
    <https://flic.kr/p/r9qvLn>`__.

    No known copyright restrictions, released into the public domain.

    Returns
    -------
    astronaut : (512, 512, 3) uint8 ndarray
        Astronaut image.
    """
    return _load('data/astronaut.png')