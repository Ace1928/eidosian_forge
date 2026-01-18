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
def light_ambient(light_color=default_color_selected, intensity=1):
    """Create a new Ambient Light
        An Ambient Light source represents an omni-directional, fixed-intensity and fixed-color light source that affects all objects in the scene equally (is omni-present).
        This light cannot be used to cast shadows.
    :param light_color: {color} Color of the Ambient Light. Default 'white'
    :param intensity: Factor used to increase or decrease the Ambient Light intensity. Default is 1
    :return: :any:`pythreejs.AmbientLight`
    """
    light = pythreejs.AmbientLight(color=light_color, intensity=intensity)
    fig = gcf()
    fig.lights = fig.lights + [light]
    return light