import os
import sys
import tokenize
def format_witnesses(w):
    firsts = (str(tup[0]) for tup in w)
    prefix = 'at tab size'
    if len(w) > 1:
        prefix = prefix + 's'
    return prefix + ' ' + ', '.join(firsts)