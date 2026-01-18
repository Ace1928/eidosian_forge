import sys
import os
import random
import logging
import json
import warnings
from numbers import Number
import numpy as np
from .. import numpy as _mx_np  # pylint: disable=reimported
from ..base import numeric_types
from .. import ndarray as nd
from ..ndarray import _internal
from .. import io
from .. import recordio
from .. util import is_np_array
from ..ndarray.numpy import _internal as _npi
def imrotate(src, rotation_degrees, zoom_in=False, zoom_out=False):
    """Rotates the input image(s) of a specific rotation degree.

    Parameters
    ----------
    src : NDArray
        Input image (format CHW) or batch of images (format NCHW),
        in both case is required a float32 data type.
    rotation_degrees: scalar or NDArray
        Wanted rotation in degrees. In case of `src` being a single image
        a scalar is needed, otherwise a mono-dimensional vector of angles
        or a scalar.
    zoom_in: bool
        If True input image(s) will be zoomed in a way so that no padding
        will be shown in the output result.
    zoom_out: bool
        If True input image(s) will be zoomed in a way so that the whole
        original image will be contained in the output result.
    Returns
    -------
    NDArray
        An `NDArray` containing the rotated image(s).
    """
    if zoom_in and zoom_out:
        raise ValueError('`zoom_in` and `zoom_out` cannot be both True')
    if src.dtype is not np.float32:
        raise TypeError('Only `float32` images are supported by this function')
    expanded = False
    if src.ndim == 3:
        expanded = True
        src = src.expand_dims(axis=0)
        if not isinstance(rotation_degrees, Number):
            raise TypeError('When a single image is passed the rotation angle is required to be a scalar.')
    elif src.ndim != 4:
        raise ValueError('Only 3D and 4D are supported by this function')
    if isinstance(rotation_degrees, Number):
        rotation_degrees = nd.array([rotation_degrees] * len(src), ctx=src.context)
    if len(src) != len(rotation_degrees):
        raise ValueError('The number of images must be equal to the number of rotation angles')
    rotation_degrees = rotation_degrees.as_in_context(src.context)
    rotation_rad = np.pi * rotation_degrees / 180
    rotation_rad = rotation_rad.expand_dims(axis=1).expand_dims(axis=2)
    _, _, h, w = src.shape
    hscale = float(h - 1) / 2
    wscale = float(w - 1) / 2
    h_matrix = (nd.repeat(nd.arange(h, ctx=src.context).astype('float32').reshape(h, 1), w, axis=1) - hscale).expand_dims(axis=0)
    w_matrix = (nd.repeat(nd.arange(w, ctx=src.context).astype('float32').reshape(1, w), h, axis=0) - wscale).expand_dims(axis=0)
    c_alpha = nd.cos(rotation_rad)
    s_alpha = nd.sin(rotation_rad)
    w_matrix_rot = w_matrix * c_alpha - h_matrix * s_alpha
    h_matrix_rot = w_matrix * s_alpha + h_matrix * c_alpha
    w_matrix_rot = w_matrix_rot / wscale
    h_matrix_rot = h_matrix_rot / hscale
    h, w = (nd.array([h], ctx=src.context), nd.array([w], ctx=src.context))
    if zoom_in or zoom_out:
        rho_corner = nd.sqrt(h * h + w * w)
        ang_corner = nd.arctan(h / w)
        corner1_x_pos = nd.abs(rho_corner * nd.cos(ang_corner + nd.abs(rotation_rad)))
        corner1_y_pos = nd.abs(rho_corner * nd.sin(ang_corner + nd.abs(rotation_rad)))
        corner2_x_pos = nd.abs(rho_corner * nd.cos(ang_corner - nd.abs(rotation_rad)))
        corner2_y_pos = nd.abs(rho_corner * nd.sin(ang_corner - nd.abs(rotation_rad)))
        max_x = nd.maximum(corner1_x_pos, corner2_x_pos)
        max_y = nd.maximum(corner1_y_pos, corner2_y_pos)
        if zoom_out:
            scale_x = max_x / w
            scale_y = max_y / h
            globalscale = nd.maximum(scale_x, scale_y)
        else:
            scale_x = w / max_x
            scale_y = h / max_y
            globalscale = nd.minimum(scale_x, scale_y)
        globalscale = globalscale.expand_dims(axis=3)
    else:
        globalscale = 1
    grid = nd.concat(w_matrix_rot.expand_dims(axis=1), h_matrix_rot.expand_dims(axis=1), dim=1)
    grid = grid * globalscale
    rot_img = nd.BilinearSampler(src, grid)
    if expanded:
        return rot_img[0]
    return rot_img