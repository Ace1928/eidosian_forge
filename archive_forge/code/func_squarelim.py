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
def squarelim():
    """Set all axes with equal aspect ratio, such that the space is 'square'."""
    fig = gcf()
    xmin, xmax = fig.xlim
    ymin, ymax = fig.ylim
    zmin, zmax = fig.zlim
    width = max([abs(xmax - xmin), abs(ymax - ymin), abs(zmax - zmin)])
    xc = (xmin + xmax) / 2
    yc = (ymin + ymax) / 2
    zc = (zmin + zmax) / 2
    xlim(xc - width / 2, xc + width / 2)
    ylim(yc - width / 2, yc + width / 2)
    zlim(zc - width / 2, zc + width / 2)