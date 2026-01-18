import bisect
def intranges_from_list(list_):
    """Represent a list of integers as a sequence of ranges:
    ((start_0, end_0), (start_1, end_1), ...), such that the original
    integers are exactly those x such that start_i <= x < end_i for some i.

    Ranges are encoded as single integers (start << 32 | end), not as tuples.
    """
    sorted_list = sorted(list_)
    ranges = []
    last_write = -1
    for i in range(len(sorted_list)):
        if i + 1 < len(sorted_list):
            if sorted_list[i] == sorted_list[i + 1] - 1:
                continue
        current_range = sorted_list[last_write + 1:i + 1]
        ranges.append(_encode_range(current_range[0], current_range[-1] + 1))
        last_write = i
    return tuple(ranges)