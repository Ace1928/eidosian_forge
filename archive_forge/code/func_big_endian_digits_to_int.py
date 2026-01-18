from typing import List, Iterable, Any, Union, Optional, overload
def big_endian_digits_to_int(digits: Iterable[int], *, base: Union[int, Iterable[int]]) -> int:
    """Returns the big-endian integer specified by the given digits and base.

    Args:
        digits: Digits of the integer, with the least significant digit at the
            end.
        base: The base, or list of per-digit bases, to use when combining the
            digits into an integer. When a list of bases is specified, the last
            entry in the list is the base for the last entry of the digits list
            (i.e. the least significant digit). That is to say, the bases are
            also specified in big endian order.

    Returns:
        The integer.

    Raises:
        ValueError:
            One of the digits is out of range for its base.
            The base was specified per-digit (as a list) but the length of the
                bases list is different from the number of digits.

    Examples:

        >>> cirq.big_endian_digits_to_int([0, 1], base=10)
        1

        >>> cirq.big_endian_digits_to_int([1, 0], base=10)
        10

        >>> cirq.big_endian_digits_to_int([1, 2, 3], base=[2, 3, 4])
        23
    """
    digits = tuple(digits)
    base = (base,) * len(digits) if isinstance(base, int) else tuple(base)
    if len(digits) != len(base):
        raise ValueError('len(digits) != len(base)')
    result = 0
    for d, b in zip(digits, base):
        if not 0 <= d < b:
            raise ValueError(f'Out of range digit. Digit: {d!r}, base: {b!r}')
        result *= b
        result += d
    return result