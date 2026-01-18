import re
import itertools
@classmethod
def maskScoreRule3hor(cls, modules, pattern=[True, False, True, True, True, False, True, False, False, False, False]):
    patternlen = len(pattern)
    score = 0
    for row in modules:
        j = 0
        maxj = len(row) - patternlen
        while j < maxj:
            if row[j:j + patternlen] == pattern:
                score += 40
                j += patternlen
            else:
                j += 1
    return score