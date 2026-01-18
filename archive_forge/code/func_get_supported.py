from __future__ import annotations
import collections
import os
import sys
import warnings
import PIL
from . import Image
def get_supported():
    """
    :returns: A list of all supported modules, features, and codecs.
    """
    ret = get_supported_modules()
    ret.extend(get_supported_features())
    ret.extend(get_supported_codecs())
    return ret