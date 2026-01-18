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
def set_box_aspect_data():
    """Sets the aspect of the bounding box equal to the aspects of the data

    For volume rendering, this makes your voxels cubes.
    """
    fig = gcf()
    xmin, xmax = fig.xlim
    ymin, ymax = fig.ylim
    zmin, zmax = fig.zlim
    size = [abs(xmax - xmin), abs(ymax - ymin), abs(zmax - zmin)]
    set_box_aspect(size)