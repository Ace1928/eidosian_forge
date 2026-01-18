import base64
from collections.abc import Sized, Sequence, Mapping
import functools
import importlib
import inspect
import io
import itertools
from numbers import Real
import re
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import matplotlib as mpl
import numpy as np
from matplotlib import _api, _cm, cbook, scale
from ._color_data import BASE_COLORS, TABLEAU_COLORS, CSS4_COLORS, XKCD_COLORS
def blend_hsv(self, rgb, intensity, hsv_max_sat=None, hsv_max_val=None, hsv_min_val=None, hsv_min_sat=None):
    """
        Take the input data array, convert to HSV values in the given colormap,
        then adjust those color values to give the impression of a shaded
        relief map with a specified light source.  RGBA values are returned,
        which can then be used to plot the shaded image with imshow.

        The color of the resulting image will be darkened by moving the (s, v)
        values (in HSV colorspace) toward (hsv_min_sat, hsv_min_val) in the
        shaded regions, or lightened by sliding (s, v) toward (hsv_max_sat,
        hsv_max_val) in regions that are illuminated.  The default extremes are
        chose so that completely shaded points are nearly black (s = 1, v = 0)
        and completely illuminated points are nearly white (s = 0, v = 1).

        Parameters
        ----------
        rgb : `~numpy.ndarray`
            An (M, N, 3) RGB array of floats ranging from 0 to 1 (color image).
        intensity : `~numpy.ndarray`
            An (M, N, 1) array of floats ranging from 0 to 1 (grayscale image).
        hsv_max_sat : number, optional
            The maximum saturation value that the *intensity* map can shift the output
            image to. If not provided, use the value provided upon initialization.
        hsv_min_sat : number, optional
            The minimum saturation value that the *intensity* map can shift the output
            image to. If not provided, use the value provided upon initialization.
        hsv_max_val : number, optional
            The maximum value ("v" in "hsv") that the *intensity* map can shift the
            output image to. If not provided, use the value provided upon
            initialization.
        hsv_min_val : number, optional
            The minimum value ("v" in "hsv") that the *intensity* map can shift the
            output image to. If not provided, use the value provided upon
            initialization.

        Returns
        -------
        `~numpy.ndarray`
            An (M, N, 3) RGB array representing the combined images.
        """
    if hsv_max_sat is None:
        hsv_max_sat = self.hsv_max_sat
    if hsv_max_val is None:
        hsv_max_val = self.hsv_max_val
    if hsv_min_sat is None:
        hsv_min_sat = self.hsv_min_sat
    if hsv_min_val is None:
        hsv_min_val = self.hsv_min_val
    intensity = intensity[..., 0]
    intensity = 2 * intensity - 1
    hsv = rgb_to_hsv(rgb[:, :, 0:3])
    hue, sat, val = np.moveaxis(hsv, -1, 0)
    np.putmask(sat, (np.abs(sat) > 1e-10) & (intensity > 0), (1 - intensity) * sat + intensity * hsv_max_sat)
    np.putmask(sat, (np.abs(sat) > 1e-10) & (intensity < 0), (1 + intensity) * sat - intensity * hsv_min_sat)
    np.putmask(val, intensity > 0, (1 - intensity) * val + intensity * hsv_max_val)
    np.putmask(val, intensity < 0, (1 + intensity) * val - intensity * hsv_min_val)
    np.clip(hsv[:, :, 1:], 0, 1, out=hsv[:, :, 1:])
    return hsv_to_rgb(hsv)