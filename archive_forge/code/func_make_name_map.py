from __future__ import absolute_import, division
import itertools
import math
import sys
import textwrap
def make_name_map(names):
    """
    Create a dictionary mapping lowercase names to capitalized names.

    Parameters
    ----------
    names : sequence

    Returns
    -------
    dict

    """
    return dict(((name.lower(), name) for name in names))