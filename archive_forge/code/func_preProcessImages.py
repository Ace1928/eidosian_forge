import os
import binascii
from io import BytesIO
from reportlab import rl_config
from reportlab.lib.utils import ImageReader, isUnicode
from reportlab.lib.rl_accel import asciiBase85Encode, asciiBase85Decode
def preProcessImages(spec):
    """Preprocesses one or more image files.

    Accepts either a filespec ('C:\\mydir\\*.jpg') or a list
    of image filenames, crunches them all to save time.  Run this
    to save huge amounts of time when repeatedly building image
    documents."""
    import glob
    if isinstance(spec, str):
        filelist = glob.glob(spec)
    else:
        filelist = spec
    for filename in filelist:
        if cachedImageExists(filename):
            if rl_config.verbose:
                print('cached version of %s already exists' % filename)
        else:
            cacheImageFile(filename)