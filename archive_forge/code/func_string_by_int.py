import string as string_module
from yaql.language import specs
from yaql.language import utils
from yaql.language import yaqltypes
@specs.parameter('left', yaqltypes.String())
@specs.parameter('right', int)
@specs.name('#operator_*')
def string_by_int(left, right, engine):
    """:yaql:operator *

    Returns string repeated count times.

    :signature: left * right
    :arg left: left operand
    :argType left: string
    :arg right: right operator, how many times repeat input string
    :argType right: integer
    :returnType: string

    .. code::

        yaql> "ab" * 2
        "abab"
    """
    utils.limit_memory_usage(engine, (-right + 1, u''), (right, left))
    return left * right