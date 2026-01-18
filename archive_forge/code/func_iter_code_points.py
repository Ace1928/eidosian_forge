from sys import maxunicode
from typing import Iterable, Iterator, Optional, Set, Tuple, Union
def iter_code_points(code_points: Iterable[CodePoint], reverse: bool=False) -> Iterator[CodePoint]:
    """
    Iterates a code points sequence. Three ore more consecutive
    code points are merged in a range.

    :param code_points: an iterable with code points and code point ranges.
    :param reverse: if `True` reverses the order of the sequence.
    :return: yields code points or code point ranges.
    """
    start_cp = end_cp = 0
    if reverse:
        code_points = sorted(code_points, key=code_point_reverse_order, reverse=True)
    else:
        code_points = sorted(code_points, key=code_point_order)
    for cp in code_points:
        if isinstance(cp, int):
            cp = (cp, cp + 1)
        if not end_cp:
            start_cp, end_cp = cp
            continue
        elif reverse:
            if start_cp <= cp[1]:
                start_cp = min(start_cp, cp[0])
                continue
        elif end_cp >= cp[0]:
            end_cp = max(end_cp, cp[1])
            continue
        if end_cp > start_cp + 1:
            yield (start_cp, end_cp)
        else:
            yield start_cp
        start_cp, end_cp = cp
    else:
        if end_cp:
            if end_cp > start_cp + 1:
                yield (start_cp, end_cp)
            else:
                yield start_cp