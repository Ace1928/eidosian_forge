import itertools
import os
import re
import sys
import textwrap
import types
from collections import OrderedDict, defaultdict
from itertools import zip_longest
from operator import itemgetter
from pprint import pprint
from nltk.corpus.reader import XMLCorpusReader, XMLCorpusView
from nltk.util import LazyConcatenation, LazyIteratorList, LazyMap
def mimic_wrap(lines, wrap_at=65, **kwargs):
    """
    Wrap the first of 'lines' with textwrap and the remaining lines at exactly the same
    positions as the first.
    """
    l0 = textwrap.fill(lines[0], wrap_at, drop_whitespace=False).split('\n')
    yield l0

    def _(line):
        il0 = 0
        while line and il0 < len(l0) - 1:
            yield line[:len(l0[il0])]
            line = line[len(l0[il0]):]
            il0 += 1
        if line:
            yield from textwrap.fill(line, wrap_at, drop_whitespace=False).split('\n')
    for l in lines[1:]:
        yield list(_(l))