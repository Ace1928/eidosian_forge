import os
import binascii
from io import BytesIO
from reportlab import rl_config
from reportlab.lib.utils import ImageReader, isUnicode
from reportlab.lib.rl_accel import asciiBase85Encode, asciiBase85Decode
def cachedImageExists(filename):
    """Determines if a cached image already exists for a given file.

    Determines if a cached image exists which has the same name
    and equal or newer date to the given file."""
    cachedname = os.path.splitext(filename)[0] + (rl_config.useA85 and '.a85' or 'bin')
    if os.path.isfile(cachedname):
        original_date = os.stat(filename)[8]
        cached_date = os.stat(cachedname)[8]
        if original_date > cached_date:
            return 0
        else:
            return 1
    else:
        return 0