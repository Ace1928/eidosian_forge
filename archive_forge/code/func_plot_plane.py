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
@_docsubst
def plot_plane(where='back', texture=None, description=None, **kwargs):
    """Plot a plane at a particular location in the viewbox.

    :param str where: 'back', 'front', 'left', 'right', 'top', 'bottom', 'x', 'y', 'z'
    :param texture: {texture}
    :param description: {description}
    :return: :any:`Mesh`
    """
    fig = gcf()
    xmin, xmax = fig.xlim
    ymin, ymax = fig.ylim
    zmin, zmax = fig.zlim
    if where == 'back':
        x = [xmin, xmax, xmax, xmin]
        y = [ymin, ymin, ymax, ymax]
        z = [zmin, zmin, zmin, zmin]
    if where == 'z':
        x = [xmin, xmax, xmax, xmin]
        y = [ymin, ymin, ymax, ymax]
        z = [0.0, 0.0, 0.0, 0.0]
    if where == 'front':
        x = [xmin, xmax, xmax, xmin][::-1]
        y = [ymin, ymin, ymax, ymax]
        z = [zmax, zmax, zmax, zmax]
    if where == 'left':
        x = [xmin, xmin, xmin, xmin]
        y = [ymin, ymin, ymax, ymax]
        z = [zmin, zmax, zmax, zmin]
    if where == 'x':
        x = [0.0, 0.0, 0.0, 0.0]
        y = [ymin, ymin, ymax, ymax]
        z = [zmin, zmax, zmax, zmin]
    if where == 'right':
        x = [xmax, xmax, xmax, xmax]
        y = [ymin, ymin, ymax, ymax]
        z = [zmin, zmax, zmax, zmin][::-1]
    if where == 'top':
        x = [xmin, xmax, xmax, xmin]
        y = [ymax, ymax, ymax, ymax]
        z = [zmax, zmax, zmin, zmin]
    if where == 'bottom':
        x = [xmax, xmin, xmin, xmax]
        y = [ymin, ymin, ymin, ymin]
        z = [zmin, zmin, zmax, zmax]
    if where == 'y':
        x = [xmax, xmin, xmin, xmax]
        y = [0.0, 0.0, 0.0, 0.0]
        z = [zmin, zmin, zmax, zmax]
    triangles = [(0, 1, 2), (0, 2, 3)]
    u = v = None
    if texture is not None:
        u = [0.0, 1.0, 1.0, 0.0]
        v = [0.0, 0.0, 1.0, 1.0]
    if description is None:
        description = f'Plane: {where}'
    mesh = plot_trisurf(x, y, z, triangles, texture=texture, u=u, v=v, description=description, **kwargs)
    return mesh