import string as string_module
from yaql.language import specs
from yaql.language import utils
from yaql.language import yaqltypes
@specs.parameter('left', int)
@specs.parameter('right', yaqltypes.String())
@specs.name('#operator_*')
def int_by_string(left, right, engine):
    """:yaql:operator *

    Returns string repeated count times.

    :signature: left * right
    :arg left: left operand, how many times repeat input string
    :argType left: integer
    :arg right: right operator
    :argType right: string
    :returnType: string

    .. code::

        yaql> 2 * "ab"
        "abab"
    """
    return string_by_int(right, left, engine)