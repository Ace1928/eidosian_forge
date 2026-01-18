import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from .miscplot import palplot
from .palettes import (color_palette, dark_palette, light_palette,
@interact
def choose_light_palette_husl(h=(0, 359), s=(0, 99), l=(0, 99), n=(3, 17)):
    color = (h, s, l)
    if as_cmap:
        colors = light_palette(color, 256, input='husl')
        _update_lut(cmap, colors)
        _show_cmap(cmap)
    else:
        pal[:] = light_palette(color, n, input='husl')
        palplot(pal)