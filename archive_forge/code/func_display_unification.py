import copy
import re
from functools import total_ordering
from nltk.internals import raise_unorderable_types, read_str
from nltk.sem.logic import (
def display_unification(fs1, fs2, indent='  '):
    fs1_lines = ('%s' % fs1).split('\n')
    fs2_lines = ('%s' % fs2).split('\n')
    if len(fs1_lines) > len(fs2_lines):
        blankline = '[' + ' ' * (len(fs2_lines[0]) - 2) + ']'
        fs2_lines += [blankline] * len(fs1_lines)
    else:
        blankline = '[' + ' ' * (len(fs1_lines[0]) - 2) + ']'
        fs1_lines += [blankline] * len(fs2_lines)
    for fs1_line, fs2_line in zip(fs1_lines, fs2_lines):
        print(indent + fs1_line + '   ' + fs2_line)
    print(indent + '-' * len(fs1_lines[0]) + '   ' + '-' * len(fs2_lines[0]))
    linelen = len(fs1_lines[0]) * 2 + 3
    print(indent + '|               |'.center(linelen))
    print(indent + '+-----UNIFY-----+'.center(linelen))
    print(indent + '|'.center(linelen))
    print(indent + 'V'.center(linelen))
    bindings = {}
    result = fs1.unify(fs2, bindings)
    if result is None:
        print(indent + '(FAILED)'.center(linelen))
    else:
        print('\n'.join((indent + l.center(linelen) for l in ('%s' % result).split('\n'))))
        if bindings and len(bindings.bound_variables()) > 0:
            print(repr(bindings).center(linelen))
    return result