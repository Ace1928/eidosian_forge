import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from .miscplot import palplot
from .palettes import (color_palette, dark_palette, light_palette,
@interact
def choose_qualitative(name=opts, n=(2, 16), desat=FloatSlider(min=0, max=1, value=1)):
    pal[:] = color_palette(name, n, desat)
    palplot(pal)