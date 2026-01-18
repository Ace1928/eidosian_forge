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
def shade_rgb(self, rgb, elevation, fraction=1.0, blend_mode='hsv', vert_exag=1, dx=1, dy=1, **kwargs):
    """
        Use this light source to adjust the colors of the *rgb* input array to
        give the impression of a shaded relief map with the given *elevation*.

        Parameters
        ----------
        rgb : array-like
            An (M, N, 3) RGB array, assumed to be in the range of 0 to 1.
        elevation : array-like
            An (M, N) array of the height values used to generate a shaded map.
        fraction : number
            Increases or decreases the contrast of the hillshade.  Values
            greater than one will cause intermediate values to move closer to
            full illumination or shadow (and clipping any values that move
            beyond 0 or 1). Note that this is not visually or mathematically
            the same as vertical exaggeration.
        blend_mode : {'hsv', 'overlay', 'soft'} or callable, optional
            The type of blending used to combine the colormapped data values
            with the illumination intensity.  For backwards compatibility, this
            defaults to "hsv". Note that for most topographic surfaces,
            "overlay" or "soft" appear more visually realistic. If a
            user-defined function is supplied, it is expected to combine an
            (M, N, 3) RGB array of floats (ranging 0 to 1) with an (M, N, 1)
            hillshade array (also 0 to 1).  (Call signature
            ``func(rgb, illum, **kwargs)``)
            Additional kwargs supplied to this function will be passed on to
            the *blend_mode* function.
        vert_exag : number, optional
            The amount to exaggerate the elevation values by when calculating
            illumination. This can be used either to correct for differences in
            units between the x-y coordinate system and the elevation
            coordinate system (e.g. decimal degrees vs. meters) or to
            exaggerate or de-emphasize topography.
        dx : number, optional
            The x-spacing (columns) of the input *elevation* grid.
        dy : number, optional
            The y-spacing (rows) of the input *elevation* grid.
        **kwargs
            Additional kwargs are passed on to the *blend_mode* function.

        Returns
        -------
        `~numpy.ndarray`
            An (m, n, 3) array of floats ranging between 0-1.
        """
    intensity = self.hillshade(elevation, vert_exag, dx, dy, fraction)
    intensity = intensity[..., np.newaxis]
    lookup = {'hsv': self.blend_hsv, 'soft': self.blend_soft_light, 'overlay': self.blend_overlay}
    if blend_mode in lookup:
        blend = lookup[blend_mode](rgb, intensity, **kwargs)
    else:
        try:
            blend = blend_mode(rgb, intensity, **kwargs)
        except TypeError as err:
            raise ValueError(f'"blend_mode" must be callable or one of {lookup.keys}') from err
    if np.ma.is_masked(intensity):
        mask = intensity.mask[..., 0]
        for i in range(3):
            blend[..., i][mask] = rgb[..., i][mask]
    return blend