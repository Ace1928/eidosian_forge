from __future__ import absolute_import
from __future__ import print_function
import webbrowser
from ..palette import Palette
from .colorbrewer_all_schemes import COLOR_MAPS
def print_all_maps():
    """
    Print the name and number of defined colors of all available color maps.

    """
    for t in MAP_TYPES:
        print_maps_by_type(t)