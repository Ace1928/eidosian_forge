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
def light_hemisphere(light_color='#ffffbb', light_color2='#080820', intensity=1, position=[0, 1, 0]):
    """Create a new Hemisphere Light

    A light source positioned directly above the scene, with color fading from the sky color to the ground color.
    This light cannot be used to cast shadows.

    :param light_color: {color} Sky color. Default white-ish 'ffffbb'.
    :param light_color2: {color} Ground color. Default greyish '#080820'
    :param intensity: Factor used to increase or decrease the Hemisphere Light intensity. Default is 1
    :param position: 3-element array (x y z) which describes the position of the Hemisphere Light. Default [0, 1, 0]
    :return: :any:`pythreejs.HemisphereLight`
    """
    light = pythreejs.HemisphereLight(color=light_color, groundColor=light_color2, intensity=intensity, position=position)
    fig = gcf()
    fig.lights = fig.lights + [light]
    return light