import logging
import threading
import numpy as np
from ..core import Format, image_as_uint
from ..core.request import URI_FILE, URI_BYTES
from .pillowmulti import GIFFormat, TIFFFormat  # noqa: E402, F401
def pil_try_read(im):
    try:
        im.getdata()[0]
    except IOError as e:
        site = 'http://pillow.readthedocs.io/en/latest/installation.html'
        site += '#external-libraries'
        pillow_error_message = str(e)
        error_message = 'Could not load "%s" \nReason: "%s"\nPlease see documentation at: %s' % (im.filename, pillow_error_message, site)
        raise ValueError(error_message)