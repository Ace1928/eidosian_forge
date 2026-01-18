from __future__ import division
import logging
import warnings
import math
from base64 import b64encode
import numpy as np
import PIL.Image
import ipywidgets
import ipywebrtc
from ipython_genutils.py3compat import string_types
from ipyvolume import utils
def create_array_cube_png_serialization(attrname, update_from_js=False):

    def fixed(value, obj=None):
        if update_from_js:
            return from_json(value)
        else:
            return getattr(obj, attrname)
    return dict(to_json=cube_to_json, from_json=fixed)