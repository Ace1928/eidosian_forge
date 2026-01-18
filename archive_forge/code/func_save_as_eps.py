from builtins import range
import sys
from math import sqrt, cos, sin, atan2, pi
def save_as_eps(self, file_name, colormode='color', width=312.0):
    """
        Save the link diagram as an encapsulated postscript file.
        Accepts options colormode and width.
        The colormode must be 'color', 'gray', or 'mono'; default is 'color'.
        The width option sets the width of the figure in points.
        The default width is 312pt = 4.33in = 11cm .
        """
    save_as_eps(self.canvas, file_name, colormode, width)