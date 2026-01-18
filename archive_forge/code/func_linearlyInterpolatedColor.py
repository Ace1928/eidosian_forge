import math, re, functools
from reportlab.lib.rl_accel import fp_str
from reportlab.lib.utils import asNative, isStr, rl_safe_eval, rl_extended_literal_eval
from reportlab import rl_config
from ast import literal_eval
import re
def linearlyInterpolatedColor(c0, c1, x0, x1, x):
    """
    Linearly interpolates colors. Can handle RGB, CMYK and PCMYK
    colors - give ValueError if colours aren't the same.
    Doesn't currently handle 'Spot Color Interpolation'.
    """
    if c0.__class__ != c1.__class__:
        raise ValueError("Color classes must be the same for interpolation!\nGot %r and %r'" % (c0, c1))
    if x1 < x0:
        x0, x1, c0, c1 = (x1, x0, c1, c0)
    if x < x0 - 1e-08 or x > x1 + 1e-08:
        raise ValueError("Can't interpolate: x=%f is not between %f and %f!" % (x, x0, x1))
    if x <= x0:
        return c0
    elif x >= x1:
        return c1
    cname = c0.__class__.__name__
    dx = float(x1 - x0)
    x = x - x0
    if cname == 'Color':
        r = c0.red + x * (c1.red - c0.red) / dx
        g = c0.green + x * (c1.green - c0.green) / dx
        b = c0.blue + x * (c1.blue - c0.blue) / dx
        a = c0.alpha + x * (c1.alpha - c0.alpha) / dx
        return Color(r, g, b, alpha=a)
    elif cname == 'CMYKColor':
        if cmykDistance(c0, c1) < 1e-08:
            assert c0.spotName == c1.spotName, 'Identical cmyk, but different spotName'
            c = c0.cyan
            m = c0.magenta
            y = c0.yellow
            k = c0.black
            d = c0.density + x * (c1.density - c0.density) / dx
            a = c0.alpha + x * (c1.alpha - c0.alpha) / dx
            return CMYKColor(c, m, y, k, density=d, spotName=c0.spotName, alpha=a)
        elif cmykDistance(c0, _CMYK_white) < 1e-08:
            c = c1.cyan
            m = c1.magenta
            y = c1.yellow
            k = c1.black
            d = x * c1.density / dx
            a = x * c1.alpha / dx
            return CMYKColor(c, m, y, k, density=d, spotName=c1.spotName, alpha=a)
        elif cmykDistance(c1, _CMYK_white) < 1e-08:
            c = c0.cyan
            m = c0.magenta
            y = c0.yellow
            k = c0.black
            d = x * c0.density / dx
            d = c0.density * (1 - x / dx)
            a = c0.alpha * (1 - x / dx)
            return PCMYKColor(c, m, y, k, density=d, spotName=c0.spotName, alpha=a)
        else:
            c = c0.cyan + x * (c1.cyan - c0.cyan) / dx
            m = c0.magenta + x * (c1.magenta - c0.magenta) / dx
            y = c0.yellow + x * (c1.yellow - c0.yellow) / dx
            k = c0.black + x * (c1.black - c0.black) / dx
            d = c0.density + x * (c1.density - c0.density) / dx
            a = c0.alpha + x * (c1.alpha - c0.alpha) / dx
            return CMYKColor(c, m, y, k, density=d, alpha=a)
    elif cname == 'PCMYKColor':
        if cmykDistance(c0, c1) < 1e-08:
            assert c0.spotName == c1.spotName, 'Identical cmyk, but different spotName'
            c = c0.cyan
            m = c0.magenta
            y = c0.yellow
            k = c0.black
            d = c0.density + x * (c1.density - c0.density) / dx
            a = c0.alpha + x * (c1.alpha - c0.alpha) / dx
            return PCMYKColor(c * 100, m * 100, y * 100, k * 100, density=d * 100, spotName=c0.spotName, alpha=100 * a)
        elif cmykDistance(c0, _CMYK_white) < 1e-08:
            c = c1.cyan
            m = c1.magenta
            y = c1.yellow
            k = c1.black
            d = x * c1.density / dx
            a = x * c1.alpha / dx
            return PCMYKColor(c * 100, m * 100, y * 100, k * 100, density=d * 100, spotName=c1.spotName, alpha=a * 100)
        elif cmykDistance(c1, _CMYK_white) < 1e-08:
            c = c0.cyan
            m = c0.magenta
            y = c0.yellow
            k = c0.black
            d = x * c0.density / dx
            d = c0.density * (1 - x / dx)
            a = c0.alpha * (1 - x / dx)
            return PCMYKColor(c * 100, m * 100, y * 100, k * 100, density=d * 100, spotName=c0.spotName, alpha=a * 100)
        else:
            c = c0.cyan + x * (c1.cyan - c0.cyan) / dx
            m = c0.magenta + x * (c1.magenta - c0.magenta) / dx
            y = c0.yellow + x * (c1.yellow - c0.yellow) / dx
            k = c0.black + x * (c1.black - c0.black) / dx
            d = c0.density + x * (c1.density - c0.density) / dx
            a = c0.alpha + x * (c1.alpha - c0.alpha) / dx
            return PCMYKColor(c * 100, m * 100, y * 100, k * 100, density=d * 100, alpha=a * 100)
    else:
        raise ValueError("Can't interpolate: Unknown color class %s!" % cname)