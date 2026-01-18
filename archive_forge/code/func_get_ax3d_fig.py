from __future__ import annotations
import importlib
import math
from functools import wraps
from string import ascii_letters
from typing import TYPE_CHECKING, Literal
import matplotlib.pyplot as plt
import numpy as np
import palettable.colorbrewer.diverging
from matplotlib import cm, colors
from pymatgen.core import Element
def get_ax3d_fig(ax: Axes=None, **kwargs) -> tuple[Axes3D, Figure]:
    """Helper function used in plot functions supporting an optional Axes3D
    argument. If ax is None, we build the `matplotlib` figure and create the
    Axes3D else we return the current active figure.

    Args:
        ax (Axes3D, optional): Axes3D object. Defaults to None.
        kwargs: keyword arguments are passed to plt.figure if ax is not None.

    Returns:
        tuple[Axes3D, Figure]: matplotlib Axes3D and corresponding figure objects
    """
    if ax is None:
        fig = plt.figure(**kwargs)
        ax = fig.add_subplot(projection='3d')
    else:
        fig = plt.gcf()
    return (ax, fig)