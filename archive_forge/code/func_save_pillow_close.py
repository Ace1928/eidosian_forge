import logging
import threading
import numpy as np
from ..core import Format, image_as_uint
from ..core.request import URI_FILE, URI_BYTES
from .pillowmulti import GIFFormat, TIFFFormat  # noqa: E402, F401
def save_pillow_close(im):
    if hasattr(im, 'close'):
        if hasattr(getattr(im, 'fp', None), 'close'):
            im.close()