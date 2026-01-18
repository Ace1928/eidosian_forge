import math
import os
import logging
from pathlib import Path
import warnings
import numpy as np
import PIL.Image
import PIL.PngImagePlugin
import matplotlib as mpl
from matplotlib import _api, cbook, cm
from matplotlib import _image
from matplotlib._image import *
import matplotlib.artist as martist
from matplotlib.backend_bases import FigureCanvasBase
import matplotlib.colors as mcolors
from matplotlib.transforms import (
def composite_images(images, renderer, magnification=1.0):
    """
    Composite a number of RGBA images into one.  The images are
    composited in the order in which they appear in the *images* list.

    Parameters
    ----------
    images : list of Images
        Each must have a `make_image` method.  For each image,
        `can_composite` should return `True`, though this is not
        enforced by this function.  Each image must have a purely
        affine transformation with no shear.

    renderer : `.RendererBase`

    magnification : float, default: 1
        The additional magnification to apply for the renderer in use.

    Returns
    -------
    image : (M, N, 4) `numpy.uint8` array
        The composited RGBA image.
    offset_x, offset_y : float
        The (left, bottom) offset where the composited image should be placed
        in the output figure.
    """
    if len(images) == 0:
        return (np.empty((0, 0, 4), dtype=np.uint8), 0, 0)
    parts = []
    bboxes = []
    for image in images:
        data, x, y, trans = image.make_image(renderer, magnification)
        if data is not None:
            x *= magnification
            y *= magnification
            parts.append((data, x, y, image._get_scalar_alpha()))
            bboxes.append(Bbox([[x, y], [x + data.shape[1], y + data.shape[0]]]))
    if len(parts) == 0:
        return (np.empty((0, 0, 4), dtype=np.uint8), 0, 0)
    bbox = Bbox.union(bboxes)
    output = np.zeros((int(bbox.height), int(bbox.width), 4), dtype=np.uint8)
    for data, x, y, alpha in parts:
        trans = Affine2D().translate(x - bbox.x0, y - bbox.y0)
        _image.resample(data, output, trans, _image.NEAREST, resample=False, alpha=alpha)
    return (output, bbox.x0 / magnification, bbox.y0 / magnification)