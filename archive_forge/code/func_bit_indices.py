def bit_indices(v):
    """Return list of indices where bits are set, 0 being the index of the least significant bit.

    >>> bit_indices(0b101)
    [0, 2]
    """
    return [i for i, b in enumerate(bin(v)[::-1]) if b == '1']