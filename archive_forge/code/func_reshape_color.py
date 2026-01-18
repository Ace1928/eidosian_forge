from __future__ import absolute_import
from __future__ import division
import pythreejs
import os
import time
import warnings
import tempfile
import uuid
import base64
from io import BytesIO as StringIO
import six
import numpy as np
import PIL.Image
import matplotlib.style
import ipywidgets
import IPython
from IPython.display import display
import ipyvolume as ipv
import ipyvolume.embed
from ipyvolume import utils
from . import ui
def reshape_color(ar):
    if dim(ar) == 4:
        return [k.reshape(-1, k.shape[-1]) for k in ar]
    else:
        return ar.reshape(-1, ar.shape[-1])