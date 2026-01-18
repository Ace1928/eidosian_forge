import random
from yaql.language import specs
from yaql.language import yaqltypes
@specs.parameter('left', yaqltypes.Number())
@specs.parameter('right', yaqltypes.Number())
@specs.name('#operator_+')
def binary_plus(left, right):
    """:yaql:operator +

    Returns the sum of left and right operands.

    :signature: left + right
    :arg left: left operand
    :argType left: number
    :arg right: right operand
    :argType right: number
    :returnType: number

    .. code::

        yaql> 3 + 2
        5
    """
    return left + right