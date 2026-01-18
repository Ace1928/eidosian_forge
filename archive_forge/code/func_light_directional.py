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
def light_directional(light_color=default_color_selected, intensity=1, position=[10, 10, 10], target=[0, 0, 0], near=0.1, far=100, shadow_camera_orthographic_size=10, cast_shadow=True):
    """Create a new Directional Light

    A Directional Light source illuminates all objects equally from a given direction.
    This light can be used to cast shadows.

    :param light_color: {color} Color of the Directional Light. Default 'white'
    :param intensity: Factor used to increase or decrease the Directional Light intensity. Default is 1
    :param position: 3-element array (x y z) which describes the position of the Directional Light. Default [10, 10, 10]
    :param target: 3-element array (x y z) which describes the target of the Directional Light. Default [0, 0, 0]
    :param cast_shadow: Property of a Directional Light to cast shadows. Default True
    :return: :any:`pythreejs.DirectionalLight`
    """
    shadow_map_size = 1024
    shadow_bias = -0.0008
    shadow_radius = 1
    camera = pythreejs.OrthographicCamera(near=near, far=far, left=-shadow_camera_orthographic_size / 2, right=shadow_camera_orthographic_size / 2, top=shadow_camera_orthographic_size / 2, bottom=-shadow_camera_orthographic_size / 2)
    shadow = pythreejs.DirectionalLightShadow(mapSize=(shadow_map_size, shadow_map_size), radius=shadow_radius, bias=shadow_bias, camera=camera)
    target = pythreejs.Object3D(position=target)
    light = pythreejs.DirectionalLight(color=light_color, intensity=intensity, position=position, target=target, castShadow=cast_shadow, shadow=shadow)
    fig = gcf()
    fig.lights = fig.lights + [light]
    return light