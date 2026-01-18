from __future__ import absolute_import
import sys
@property
def mpl_colors(self):
    """
        Colors expressed on the range 0-1 as used by matplotlib.

        """
    mc = []
    for color in self.colors:
        mc.append(tuple([x / 255.0 for x in color]))
    return mc