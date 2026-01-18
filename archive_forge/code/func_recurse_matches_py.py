import difflib
from bisect import bisect
from typing import Any, Dict, List, Optional, Sequence, Tuple
def recurse_matches_py(a: Sequence[Any], b: Sequence[Any], alo: int, blo: int, ahi: int, bhi: int, answer: List[Tuple[int, int]], maxrecursion: int) -> None:
    """Find all of the matching text in the lines of a and b.

    :param a: A sequence
    :param b: Another sequence
    :param alo: The start location of a to check, typically 0
    :param ahi: The start location of b to check, typically 0
    :param ahi: The maximum length of a to check, typically len(a)
    :param bhi: The maximum length of b to check, typically len(b)
    :param answer: The return array. Will be filled with tuples
                   indicating [(line_in_a, line_in_b)]
    :param maxrecursion: The maximum depth to recurse.
                         Must be a positive integer.
    :return: None, the return value is in the parameter answer, which
             should be a list

    """
    if maxrecursion < 0:
        raise MaxRecursionDepth()
    oldlength = len(answer)
    if alo == ahi or blo == bhi:
        return
    last_a_pos = alo - 1
    last_b_pos = blo - 1
    for apos, bpos in unique_lcs_py(a[alo:ahi], b[blo:bhi]):
        apos += alo
        bpos += blo
        if last_a_pos + 1 != apos or last_b_pos + 1 != bpos:
            recurse_matches_py(a, b, last_a_pos + 1, last_b_pos + 1, apos, bpos, answer, maxrecursion - 1)
        last_a_pos = apos
        last_b_pos = bpos
        answer.append((apos, bpos))
    if len(answer) > oldlength:
        recurse_matches_py(a, b, last_a_pos + 1, last_b_pos + 1, ahi, bhi, answer, maxrecursion - 1)
    elif a[alo] == b[blo]:
        while alo < ahi and blo < bhi and (a[alo] == b[blo]):
            answer.append((alo, blo))
            alo += 1
            blo += 1
        recurse_matches_py(a, b, alo, blo, ahi, bhi, answer, maxrecursion - 1)
    elif a[ahi - 1] == b[bhi - 1]:
        nahi = ahi - 1
        nbhi = bhi - 1
        while nahi > alo and nbhi > blo and (a[nahi - 1] == b[nbhi - 1]):
            nahi -= 1
            nbhi -= 1
        recurse_matches_py(a, b, last_a_pos + 1, last_b_pos + 1, nahi, nbhi, answer, maxrecursion - 1)
        for i in range(ahi - nahi):
            answer.append((nahi + i, nbhi + i))