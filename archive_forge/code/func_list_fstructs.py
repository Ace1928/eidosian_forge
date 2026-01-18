import copy
import re
from functools import total_ordering
from nltk.internals import raise_unorderable_types, read_str
from nltk.sem.logic import (
def list_fstructs(fstructs):
    for i, fstruct in fstructs:
        print()
        lines = ('%s' % fstruct).split('\n')
        print('%3d: %s' % (i + 1, lines[0]))
        for line in lines[1:]:
            print('     ' + line)
    print()