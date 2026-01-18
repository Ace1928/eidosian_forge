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
def light_point(light_color=default_color_selected, intensity=1, position=[10, 10, 10], shadow_map_size=1024, cast_shadow=True):
    """Create a new Point Light

    A Point Light originates from a single point and spreads outward in all directions.
    This light can be used to cast shadows.

    :param light_color: {color} Color of the Point Light. Default 'white'
    :param intensity: Factor used to increase or decrease the Point Light intensity. Default is 1
    :param position: 3-element array (x y z) which describes the position of the Point Light. Default [0 1 0]
    :param cast_shadow: Property of a Point Light to cast shadows. Default False
    :return: :any:`PointLight`
    """
    near = 0.1
    far = 100
    fov = 90
    aspect = 1
    distance = 0
    decay = 1
    shadow_bias = -0.0008
    shadow_radius = 1
    camera = pythreejs.PerspectiveCamera(near=near, far=far, fov=fov, aspect=aspect)
    shadow = pythreejs.LightShadow(mapSize=(shadow_map_size, shadow_map_size), radius=shadow_radius, bias=shadow_bias, camera=camera)
    light = pythreejs.PointLight(color=light_color, intensity=intensity, position=position, distance=distance, decay=decay, castShadow=cast_shadow, shadow=shadow)
    fig = gcf()
    fig.lights = fig.lights + [light]
    return light