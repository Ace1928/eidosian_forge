import colorsys
from itertools import cycle
import numpy as np
import matplotlib as mpl
from .external import husl
from .utils import desaturate, get_color_cycle
from .colors import xkcd_rgb, crayons
from ._compat import get_colormap
def set_color_codes(palette='deep'):
    """Change how matplotlib color shorthands are interpreted.

    Calling this will change how shorthand codes like "b" or "g"
    are interpreted by matplotlib in subsequent plots.

    Parameters
    ----------
    palette : {deep, muted, pastel, dark, bright, colorblind}
        Named seaborn palette to use as the source of colors.

    See Also
    --------
    set : Color codes can be set through the high-level seaborn style
          manager.
    set_palette : Color codes can also be set through the function that
                  sets the matplotlib color cycle.

    """
    if palette == 'reset':
        colors = [(0.0, 0.0, 1.0), (0.0, 0.5, 0.0), (1.0, 0.0, 0.0), (0.75, 0.0, 0.75), (0.75, 0.75, 0.0), (0.0, 0.75, 0.75), (0.0, 0.0, 0.0)]
    elif not isinstance(palette, str):
        err = 'set_color_codes requires a named seaborn palette'
        raise TypeError(err)
    elif palette in SEABORN_PALETTES:
        if not palette.endswith('6'):
            palette = palette + '6'
        colors = SEABORN_PALETTES[palette] + [(0.1, 0.1, 0.1)]
    else:
        err = f"Cannot set colors with palette '{palette}'"
        raise ValueError(err)
    for code, color in zip('bgrmyck', colors):
        rgb = mpl.colors.colorConverter.to_rgb(color)
        mpl.colors.colorConverter.colors[code] = rgb