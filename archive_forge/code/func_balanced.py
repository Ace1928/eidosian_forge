import itertools
import numpy as np
from patsy import PatsyError
from patsy.categorical import C
from patsy.util import no_pickling, assert_no_pickling
def balanced(**kwargs):
    """balanced(factor_name=num_levels, [factor_name=num_levels, ..., repeat=1])

    Create simple balanced factorial designs for testing.

    Given some factor names and the number of desired levels for each,
    generates a balanced factorial design in the form of a data
    dictionary. For example:

    .. ipython::

       In [1]: balanced(a=2, b=3)
       Out[1]:
       {'a': ['a1', 'a1', 'a1', 'a2', 'a2', 'a2'],
        'b': ['b1', 'b2', 'b3', 'b1', 'b2', 'b3']}

    By default it produces exactly one instance of each combination of levels,
    but if you want multiple replicates this can be accomplished via the
    `repeat` argument:

    .. ipython::

       In [2]: balanced(a=2, b=2, repeat=2)
       Out[2]:
       {'a': ['a1', 'a1', 'a2', 'a2', 'a1', 'a1', 'a2', 'a2'],
        'b': ['b1', 'b2', 'b1', 'b2', 'b1', 'b2', 'b1', 'b2']}
    """
    repeat = kwargs.pop('repeat', 1)
    levels = []
    names = sorted(kwargs)
    for name in names:
        level_count = kwargs[name]
        levels.append(['%s%s' % (name, i) for i in range(1, level_count + 1)])
    values = zip(*itertools.product(*levels))
    data = {}
    for name, value in zip(names, values):
        data[name] = list(value) * repeat
    return data