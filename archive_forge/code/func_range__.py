import collections
import functools
import itertools
from yaql.language import exceptions
from yaql.language import specs
from yaql.language import utils
from yaql.language import yaqltypes
@specs.parameter('start', int)
@specs.parameter('stop', int)
@specs.parameter('step', int)
def range__(start, stop, step=1):
    """:yaql:range

    Returns an iterator over values from start up to stop, not including stop,
    i.e [start, stop) with step 1 if not specified.

    :signature: range(start, stop, step => 1)
    :arg start: left bound for generated list numbers
    :argType start: integer
    :arg stop: right bound for generated list numbers
    :argType stop: integer
    :arg step: the next element in list is equal to previous + step. 1 is value
        by default
    :argType step: integer
    :returnType: iterator

    .. code::

        yaql> range(1, 4)
        [1, 2, 3]
        yaql> range(4, 1, -1)
        [4, 3, 2]
    """
    return iter(range(start, stop, step))