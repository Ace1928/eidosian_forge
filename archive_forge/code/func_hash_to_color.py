import itertools
import functools
import importlib.util
def hash_to_color(s, hmin=0.0, hmax=1.0, smin=0.3, smax=0.8, vmin=0.8, vmax=1.0):
    """Generate a random color for a string  ``s``.

    Parameters
    ----------
    s : str
        The string to generate a color for.
    hmin : float, optional
        The minimum hue value.
    hmax : float, optional
        The maximum hue value.
    smin : float, optional
        The minimum saturation value.
    smax : float, optional
        The maximum saturation value.
    vmin : float, optional
        The minimum value value.
    vmax : float, optional
        The maximum value value.

    Returns
    -------
    color : tuple
        A tuple of floats in the range [0, 1] representing the RGB color.
    """
    from matplotlib.colors import hsv_to_rgb
    h, s, v = hash_to_nvalues(s, 3)
    h = hmin + h * (hmax - hmin)
    s = smin + s * (smax - smin)
    v = vmin + v * (vmax - vmin)
    return hsv_to_rgb((h, s, v))