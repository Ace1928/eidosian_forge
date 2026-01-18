import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from .miscplot import palplot
from .palettes import (color_palette, dark_palette, light_palette,
@interact
def choose_diverging_palette(h_neg=IntSlider(min=0, max=359, value=220), h_pos=IntSlider(min=0, max=359, value=10), s=IntSlider(min=0, max=99, value=74), l=IntSlider(min=0, max=99, value=50), sep=IntSlider(min=1, max=50, value=10), n=(2, 16), center=['light', 'dark']):
    if as_cmap:
        colors = diverging_palette(h_neg, h_pos, s, l, sep, 256, center)
        _update_lut(cmap, colors)
        _show_cmap(cmap)
    else:
        pal[:] = diverging_palette(h_neg, h_pos, s, l, sep, n, center)
        palplot(pal)