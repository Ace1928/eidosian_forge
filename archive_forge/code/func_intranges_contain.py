import bisect
def intranges_contain(int_, ranges):
    """Determine if `int_` falls into one of the ranges in `ranges`."""
    tuple_ = _encode_range(int_, 0)
    pos = bisect.bisect_left(ranges, tuple_)
    if pos > 0:
        left, right = _decode_range(ranges[pos - 1])
        if left <= int_ < right:
            return True
    if pos < len(ranges):
        left, _ = _decode_range(ranges[pos])
        if left == int_:
            return True
    return False