import os
def damerau_levenshtein(s1, s2, cost):
    """Calculates the Damerau-Levenshtein distance between two strings.

    The Damerau-Levenshtein distance says the minimum number of single
    character edits (i.e. insertions, deletions, swap or substitution)
    required to change one string to the other.
    The idea is to reserve a matrix to hold the Levenshtein distances between
    all prefixes of the first string and all prefixes of the second, then we
    can compute the values in the matrix in a dynamic programming fashion. To
    avoid a large space complexity, only the last three rows in the matrix is
    needed.(row2 holds the current row, row1 holds the previous row, and row0
    the row before that.)

    More details:
        https://en.wikipedia.org/wiki/Damerau%E2%80%93Levenshtein_distance
        https://github.com/git/git/commit/8af84dadb142f7321ff0ce8690385e99da8ede2f
    """
    if s1 == s2:
        return 0
    len1 = len(s1)
    len2 = len(s2)
    if len1 == 0:
        return len2 * cost['a']
    if len2 == 0:
        return len1 * cost['d']
    row1 = [i * cost['a'] for i in range(len2 + 1)]
    row2 = row1[:]
    row0 = row1[:]
    for i in range(len1):
        row2[0] = (i + 1) * cost['d']
        for j in range(len2):
            sub_cost = row1[j] + (s1[i] != s2[j]) * cost['s']
            ins_cost = row2[j] + cost['a']
            del_cost = row1[j + 1] + cost['d']
            swp_condition = i > 0 and j > 0 and (s1[i - 1] == s2[j]) and (s1[i] == s2[j - 1])
            if swp_condition:
                swp_cost = row0[j - 1] + cost['w']
                p_cost = min(sub_cost, ins_cost, del_cost, swp_cost)
            else:
                p_cost = min(sub_cost, ins_cost, del_cost)
            row2[j + 1] = p_cost
        row0, row1, row2 = (row1, row2, row0)
    return row1[-1]