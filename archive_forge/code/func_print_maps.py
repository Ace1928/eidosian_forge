from __future__ import absolute_import, print_function
from ..palette import Palette
def print_maps():
    """
    Print a list of Tableau palettes.

    """
    namelen = max((len(name) for name in palette_names))
    fmt = '{0:' + str(namelen + 4) + '}{1:16}{2:}'
    for i, name in enumerate(palette_names):
        print(fmt.format(name, palette_type, len(colors_rgb[i])))