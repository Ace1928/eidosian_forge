import operator
import warnings
def edit_distance_align(s1, s2, substitution_cost=1):
    """
    Calculate the minimum Levenshtein edit-distance based alignment
    mapping between two strings. The alignment finds the mapping
    from string s1 to s2 that minimizes the edit distance cost.
    For example, mapping "rain" to "shine" would involve 2
    substitutions, 2 matches and an insertion resulting in
    the following mapping:
    [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (4, 5)]
    NB: (0, 0) is the start state without any letters associated
    See more: https://web.stanford.edu/class/cs124/lec/med.pdf

    In case of multiple valid minimum-distance alignments, the
    backtrace has the following operation precedence:

    1. Substitute s1 and s2 characters
    2. Skip s1 character
    3. Skip s2 character

    The backtrace is carried out in reverse string order.

    This function does not support transposition.

    :param s1, s2: The strings to be aligned
    :type s1: str
    :type s2: str
    :type substitution_cost: int
    :rtype: List[Tuple(int, int)]
    """
    len1 = len(s1)
    len2 = len(s2)
    lev = _edit_dist_init(len1 + 1, len2 + 1)
    for i in range(len1):
        for j in range(len2):
            _edit_dist_step(lev, i + 1, j + 1, s1, s2, 0, 0, substitution_cost=substitution_cost, transpositions=False)
    alignment = _edit_dist_backtrace(lev)
    return alignment