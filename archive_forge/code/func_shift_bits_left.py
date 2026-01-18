import random
from yaql.language import specs
from yaql.language import yaqltypes
@specs.parameter('value', int)
@specs.parameter('bits_number', int)
def shift_bits_left(value, bits_number):
    """:yaql:shiftBitsLeft

    Shifts the bits of value left by the number of bits bitsNumber.

    :signature: shiftBitsLeft(value, bitsNumber)
    :arg value: given value
    :argType value: integer
    :arg bitsNumber: number of bits
    :argType right: integer
    :returnType: integer

    .. code::

        yaql> shiftBitsLeft(8, 2)
        32
    """
    return value << bits_number