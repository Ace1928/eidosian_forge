import random
from yaql.language import specs
from yaql.language import yaqltypes
@specs.parameter('a', yaqltypes.Number())
@specs.parameter('b', yaqltypes.Number())
@specs.parameter('c', yaqltypes.Number(nullable=True))
def pow_(a, b, c=None):
    """:yaql:pow

    Returns a to the power b modulo c.

    :signature: pow(a, b, c => null)
    :arg a: input value
    :argType a: number
    :arg b: power
    :argType b: number
    :arg c: modulo. null by default, which means no modulo is done after power.
    :argType c: integer
    :returnType: number

    .. code::

        yaql> pow(3, 2)
        9
        yaql> pow(3, 2, 5)
        4
    """
    return pow(a, b, c)