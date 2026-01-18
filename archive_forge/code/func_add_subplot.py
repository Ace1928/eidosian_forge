import base64
import json
import warnings
from binascii import hexlify
from collections import OrderedDict
from html import escape
from os import urandom
from pathlib import Path
from urllib.request import urlopen
from jinja2 import Environment, PackageLoader, Template
from .utilities import _camelify, _parse_size, none_max, none_min
def add_subplot(self, x, y, n, margin=0.05):
    """Creates a div child subplot in a matplotlib.figure.add_subplot style.

        Parameters
        ----------
        x : int
            The number of rows in the grid.
        y : int
            The number of columns in the grid.
        n : int
            The cell number in the grid, counted from 1 to x*y.

        Example:
        >>> fig.add_subplot(3, 2, 5)
        # Create a div in the 5th cell of a 3rows x 2columns
        grid(bottom-left corner).
        """
    width = 1.0 / y
    height = 1.0 / x
    left = (n - 1) % y * width
    top = (n - 1) // y * height
    left = left + width * margin
    top = top + height * margin
    width = width * (1 - 2.0 * margin)
    height = height * (1 - 2.0 * margin)
    div = Div(position='absolute', width=f'{100.0 * width}%', height=f'{100.0 * height}%', left=f'{100.0 * left}%', top=f'{100.0 * top}%')
    self.add_child(div)
    return div