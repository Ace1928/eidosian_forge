from __future__ import absolute_import, print_function, division
import hashlib
import random as pyrandom
import time
from collections import OrderedDict
from functools import partial
from petl.compat import xrange, text_type
from petl.util.base import Table
def dummytable(numrows=100, fields=(('foo', partial(pyrandom.randint, 0, 100)), ('bar', partial(pyrandom.choice, ('apples', 'pears', 'bananas', 'oranges'))), ('baz', pyrandom.random)), wait=0, seed=None):
    """
    Construct a table with dummy data. Use `numrows` to specify the number of
    rows. Set `wait` to a float greater than zero to simulate a delay on each
    row generation (number of seconds per row). E.g.::

        >>> import petl as etl
        >>> table1 = etl.dummytable(100, seed=42)
        >>> table1
        +-----+----------+----------------------+
        | foo | bar      | baz                  |
        +=====+==========+======================+
        |  81 | 'apples' | 0.025010755222666936 |
        +-----+----------+----------------------+
        |  35 | 'pears'  |  0.22321073814882275 |
        +-----+----------+----------------------+
        |  94 | 'apples' |   0.6766994874229113 |
        +-----+----------+----------------------+
        |  69 | 'apples' |   0.5904925124490397 |
        +-----+----------+----------------------+
        |   4 | 'apples' |  0.09369523986159245 |
        +-----+----------+----------------------+
        ...
        <BLANKLINE>

        >>> import random as pyrandom
        >>> from functools import partial
        >>> fields = [('foo', pyrandom.random),
        ...           ('bar', partial(pyrandom.randint, 0, 500)),
        ...           ('baz', partial(pyrandom.choice, ['chocolate', 'strawberry', 'vanilla']))]
        >>> table2 = etl.dummytable(100, fields=fields, seed=42)
        >>> table2
        +---------------------+-----+-------------+
        | foo                 | bar | baz         |
        +=====================+=====+=============+
        |  0.6394267984578837 |  12 | 'vanilla'   |
        +---------------------+-----+-------------+
        | 0.27502931836911926 | 114 | 'chocolate' |
        +---------------------+-----+-------------+
        |  0.7364712141640124 | 346 | 'vanilla'   |
        +---------------------+-----+-------------+
        |  0.8921795677048454 |  44 | 'vanilla'   |
        +---------------------+-----+-------------+
        |  0.4219218196852704 |  15 | 'chocolate' |
        +---------------------+-----+-------------+
        ...
        <BLANKLINE>

        >>> table3_1 = etl.dummytable(50)
        >>> table3_2 = etl.dummytable(100)
        >>> table3_1[5] == table3_2[5]
        False

    Data generation functions can be specified via the `fields` keyword
    argument.

    Note that the data are generated on the fly and are not stored in memory,
    so this function can be used to simulate very large tables.
    The only supported seed types are: None, int, float, str, bytes, and bytearray.

    """
    return DummyTable(numrows=numrows, fields=fields, wait=wait, seed=seed)