from typing import List, Iterable, Any, Union, Optional, overload
def big_endian_int_to_digits(val: int, *, digit_count: Optional[int]=None, base: Union[int, Iterable[int]]) -> List[int]:
    """Separates an integer into big-endian digits.

    Args:
        val: The integer to get digits from. Must be non-negative and less than
            the maximum representable value, given the specified base(s) and
            digit count.
        base: The base, or list of per-digit bases, to separate `val` into. When
             a list of bases is specified, the last entry in the list is the
             base for the last entry of the result (i.e. the least significant
             digit). That is to say, the bases are also specified in big endian
             order.
        digit_count: The length of the desired result.

    Returns:
        The list of digits.

    Raises:
        ValueError:
            Unknown digit count. The `base` was specified as an integer and a
                `digit_count` was not provided.
            Inconsistent digit count. The `base` was specified as a per-digit
                list, and `digit_count` was also provided, but they disagree.

    Examples:
        >>> cirq.big_endian_int_to_digits(11, digit_count=4, base=10)
        [0, 0, 1, 1]

        >>> cirq.big_endian_int_to_digits(11, base=[2, 3, 4])
        [0, 2, 3]
    """
    if isinstance(base, int):
        if digit_count is None:
            raise ValueError('No digit count. Provide `digit_count` when base is an int.')
        base = (base,) * digit_count
    else:
        base = tuple(base)
        if digit_count is None:
            digit_count = len(base)
    if len(base) != digit_count:
        raise ValueError('Inconsistent digit count. len(base) != digit_count')
    result = []
    for b in reversed(base):
        result.append(val % b)
        val //= b
    if val:
        raise ValueError(f'Out of range. Extracted digits {result!r} but the long division process left behind {val!r} instead of 0.')
    return result[::-1]