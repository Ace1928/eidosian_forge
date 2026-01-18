from abc import ABCMeta, abstractmethod
from nltk import jsontags
def range_to_str(positions):
    if len(positions) == 1:
        p = positions[0]
        if p == 0:
            return 'this word'
        if p == -1:
            return 'the preceding word'
        elif p == 1:
            return 'the following word'
        elif p < 0:
            return 'word i-%d' % -p
        elif p > 0:
            return 'word i+%d' % p
    else:
        mx = max(positions)
        mn = min(positions)
        if mx - mn == len(positions) - 1:
            return 'words i%+d...i%+d' % (mn, mx)
        else:
            return 'words {{{}}}'.format(','.join(('i%+d' % d for d in positions)))