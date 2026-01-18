from typing import List, Iterable, Any, Union, Optional, overload
def big_endian_bits_to_int(bits: Iterable[Any]) -> int:
    """Returns the big-endian integer specified by the given bits.

    Args:
        bits: Descending bits of the integer, with the 1s bit at the end.

    Returns:
        The integer.

    Examples:

        >>> cirq.big_endian_bits_to_int([0, 1])
        1

        >>> cirq.big_endian_bits_to_int([1, 0])
        2

        >>> cirq.big_endian_bits_to_int([0, 1, 0])
        2

        >>> cirq.big_endian_bits_to_int([1, 0, 0, 1, 0])
        18
    """
    result = 0
    for e in bits:
        result <<= 1
        if e:
            result |= 1
    return result