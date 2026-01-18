from typing import List, Iterable, Any, Union, Optional, overload
def big_endian_int_to_bits(val: int, *, bit_count: int) -> List[int]:
    """Returns the big-endian bits of an integer.

    Args:
        val: The integer to get bits from. This integer is permitted to be
            larger than `2**bit_count` (in which case the high bits of the
            result are dropped) or to be negative (in which case the bits come
            from the 2s complement signed representation).
        bit_count: The number of desired bits in the result.

    Returns:
        The bits.

    Examples:
        >>> cirq.big_endian_int_to_bits(19, bit_count=8)
        [0, 0, 0, 1, 0, 0, 1, 1]

        >>> cirq.big_endian_int_to_bits(19, bit_count=4)
        [0, 0, 1, 1]

        >>> cirq.big_endian_int_to_bits(-3, bit_count=4)
        [1, 1, 0, 1]
    """
    return [val >> i & 1 for i in range(bit_count)[::-1]]