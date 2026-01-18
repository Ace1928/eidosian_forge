import os
from glob import glob
import re
from collections.abc import Sequence
from copy import copy
import numpy as np
from PIL import Image
from tifffile import TiffFile
def alphanumeric_key(s):
    """Convert string to list of strings and ints that gives intuitive sorting.

    Parameters
    ----------
    s : string

    Returns
    -------
    k : a list of strings and ints

    Examples
    --------
    >>> alphanumeric_key('z23a')
    ['z', 23, 'a']
    >>> filenames = ['f9.10.png', 'e10.png', 'f9.9.png', 'f10.10.png',
    ...              'f10.9.png']
    >>> sorted(filenames)
    ['e10.png', 'f10.10.png', 'f10.9.png', 'f9.10.png', 'f9.9.png']
    >>> sorted(filenames, key=alphanumeric_key)
    ['e10.png', 'f9.9.png', 'f9.10.png', 'f10.9.png', 'f10.10.png']
    """
    k = [int(c) if c.isdigit() else c for c in re.split('([0-9]+)', s)]
    return k