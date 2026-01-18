import random
from yaql.language import specs
from yaql.language import yaqltypes
def random__(from_, to_):
    """:yaql:random

    Returns the next random integer from [a, b].

    :signature: random(from, to)
    :arg from: left value for generating random number
    :argType from: integer
    :arg to: right value for generating random number
    :argType to: integer
    :returnType: integer

    .. code::

        yaql> random(1, 2)
        2
        yaql> random(1, 2)
        1
    """
    return random.randint(from_, to_)