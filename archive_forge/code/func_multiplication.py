import random
from yaql.language import specs
from yaql.language import yaqltypes
@specs.parameter('left', yaqltypes.Number())
@specs.parameter('right', yaqltypes.Number())
@specs.name('#operator_*')
def multiplication(left, right):
    """:yaql:operator *

    Returns left multiplied by right.

    :signature: left * right
    :arg left: left operand
    :argType left: number
    :arg right: right operand
    :argType right: number
    :returnType: number

    .. code::

        yaql> 3 * 2.5
        7.5
    """
    return left * right