from collections.abc import Callable, Sequence
from os import listdir, path
import numpy as np
from .functions import clip_array, clip_scalar, colorDistance, eq, mkColor
from .Qt import QtCore, QtGui
def makeHslCycle(hue=0.0, saturation=1.0, lightness=0.5, steps=36):
    """
    Returns a ColorMap object that traces a circular or spiraling path around the HSL color space.

    Parameters
    ----------
    hue : float or tuple of floats
        Starting point or (start, end) for hue. Values can lie outside the [0 to 1] range 
        to realize multiple cycles. For a single value, one full hue cycle is generated.
        The default starting hue is 0.0 (red). 
    saturation : float or tuple of floats, optional
        Saturation value for the colors in the cycle, in the range of [0 to 1]. 
        If a (start, end) tuple is given, saturation gradually changes between these values.
        The default saturation is 1.0.
    lightness : float or tuple of floats, optional
        Lightness value for the colors in the cycle, in the range of [0 to 1]. 
        If a (start, end) tuple is given, lightness gradually changes between these values.
        The default lightness is 0.5.
    steps: int, optional
        Number of steps in the cycle. Between these steps, the color map will interpolate in RGB space.
        The default number of steps is 36, generating a color map with 37 stops.
    """
    if isinstance(hue, (tuple, list)):
        hueA, hueB = hue
    else:
        hueA = hue
        hueB = hueA + 1.0
    if isinstance(saturation, (tuple, list)):
        satA, satB = saturation
    else:
        satA = satB = saturation
    if isinstance(lightness, (tuple, list)):
        lgtA, lgtB = lightness
    else:
        lgtA = lgtB = lightness
    hue_vals = np.linspace(hueA, hueB, num=steps + 1)
    sat_vals = np.linspace(satA, satB, num=steps + 1)
    lgt_vals = np.linspace(lgtA, lgtB, num=steps + 1)
    color_list = []
    for hue, sat, lgt in zip(hue_vals, sat_vals, lgt_vals):
        qcol = QtGui.QColor.fromHslF(hue % 1.0, sat, lgt)
        color_list.append(qcol)
    name = f'Hue {hueA:0.2f}-{hueB:0.2f}'
    return ColorMap(None, color_list, name=name)