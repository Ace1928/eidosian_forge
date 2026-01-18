from __future__ import absolute_import, print_function, division
import hashlib
import random as pyrandom
import time
from collections import OrderedDict
from functools import partial
from petl.compat import xrange, text_type
from petl.util.base import Table
def randomtable(numflds=5, numrows=100, wait=0, seed=None):
    """
    Construct a table with random numerical data. Use `numflds` and `numrows` to
    specify the number of fields and rows respectively. Set `wait` to a float
    greater than zero to simulate a delay on each row generation (number of
    seconds per row). E.g.::

        >>> import petl as etl
        >>> table = etl.randomtable(3, 100, seed=42)
        >>> table
        +----------------------+----------------------+---------------------+
        | f0                   | f1                   | f2                  |
        +======================+======================+=====================+
        |   0.6394267984578837 | 0.025010755222666936 | 0.27502931836911926 |
        +----------------------+----------------------+---------------------+
        |  0.22321073814882275 |   0.7364712141640124 |  0.6766994874229113 |
        +----------------------+----------------------+---------------------+
        |   0.8921795677048454 |  0.08693883262941615 |  0.4219218196852704 |
        +----------------------+----------------------+---------------------+
        | 0.029797219438070344 |  0.21863797480360336 |  0.5053552881033624 |
        +----------------------+----------------------+---------------------+
        | 0.026535969683863625 |   0.1988376506866485 |  0.6498844377795232 |
        +----------------------+----------------------+---------------------+
        ...
        <BLANKLINE>

    Note that the data are generated on the fly and are not stored in memory,
    so this function can be used to simulate very large tables.
    The only supported seed types are: None, int, float, str, bytes, and bytearray.

    """
    return RandomTable(numflds, numrows, wait=wait, seed=seed)