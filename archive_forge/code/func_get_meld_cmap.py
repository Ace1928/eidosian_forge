import pandas as pd
import graphtools.base
import graphtools
import pygsp
import scprep
import sklearn
def get_meld_cmap():
    """Returns cmap used in publication for displaying EES.
    Inspired by cmocean `balance` cmap"""
    base_colors = [[0.22107637, 0.53245276, 0.72819301, 1.0], [0.7, 0.7, 0.7, 1], [0.75013244, 0.3420382, 0.22753009, 1.0]]
    return scprep.plot.tools.create_colormap(base_colors)