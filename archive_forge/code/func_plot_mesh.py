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
def plot_mesh(x, y, z, color=default_color, wireframe=True, surface=True, wrapx=False, wrapy=False, u=None, v=None, texture=None, cast_shadow=True, receive_shadow=True, description=None):
    """Draws a 2d wireframe+surface in 3d: generalization of :any:`plot_wireframe` and :any:`plot_surface`.

    :param x: {x2d}
    :param y: {y2d}
    :param z: {z2d}
    :param color: {color2d}
    :param bool wireframe: draw lines between the vertices
    :param bool surface: draw faces/triangles between the vertices
    :param bool wrapx: when True, the x direction is assumed to wrap, and polygons are drawn between the begin and end points
    :param boool wrapy: idem for y
    :param u: {u}
    :param v: {v}
    :param texture: {texture}
    :param cast_shadow: {cast_shadow}
    :param receive_shadow: {receive_shadow}
    :param description: {description}
    :return: :any:`Mesh`
    """
    fig = gcf()

    def dim(x):
        d = 0
        el = x
        while True:
            try:
                el = el[0]
                d += 1
            except:
                break
        return d
    if dim(x) == 2:
        nx, ny = x.shape
    else:
        nx, ny = x[0].shape

    def reshape(ar):
        if dim(ar) == 3:
            return [k.reshape(-1) for k in ar]
        else:
            return ar.reshape(-1)
    x = reshape(x)
    y = reshape(y)
    z = reshape(z)
    if u is not None:
        u = reshape(u)
    if v is not None:
        v = reshape(v)

    def reshape_color(ar):
        if dim(ar) == 4:
            return [k.reshape(-1, k.shape[-1]) for k in ar]
        else:
            return ar.reshape(-1, ar.shape[-1])
    if isinstance(color, np.ndarray):
        color = reshape_color(color)
    _grow_limits(np.array(x).reshape(-1), np.array(y).reshape(-1), np.array(z).reshape(-1))
    triangles, lines = _make_triangles_lines((nx, ny), wrapx, wrapy)
    kwargs = {}
    if current.material is not None:
        kwargs['material'] = current.material
    mesh = ipv.Mesh(x=x, y=y, z=z, triangles=triangles if surface else None, color=color, lines=lines if wireframe else None, u=u, v=v, texture=texture, cast_shadow=cast_shadow, receive_shadow=receive_shadow, description=f'Mesh {len(fig.meshes)}' if description is None else description, **kwargs)
    fig.meshes = fig.meshes + [mesh]
    return mesh