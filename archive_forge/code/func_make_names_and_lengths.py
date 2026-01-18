from __future__ import absolute_import, division
import itertools
import math
import sys
import textwrap
def make_names_and_lengths(names, lengths=None):
    """
    Create a list pairing palette names with lengths. (Mostly used to define
    the set of palettes that are automatically built.)

    Parameters
    ----------
    names : sequence of str
    lengths : sequence of int, optional

    Returns
    -------
    list of tuple
        Pairs of names and lengths.

    """
    lengths = lengths or range(3, 21)
    return list(itertools.product(names, lengths))