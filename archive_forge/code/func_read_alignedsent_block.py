import bisect
import os
import pickle
import re
import tempfile
from functools import reduce
from xml.etree import ElementTree
from nltk.data import (
from nltk.internals import slice_bounds
from nltk.tokenize import wordpunct_tokenize
from nltk.util import AbstractLazySequence, LazyConcatenation, LazySubsequence
def read_alignedsent_block(stream):
    s = ''
    while True:
        line = stream.readline()
        if line[0] == '=' or line[0] == '\n' or line[:2] == '\r\n':
            continue
        if not line:
            if s:
                return [s]
            else:
                return []
        else:
            s += line
            if re.match('^\\d+-\\d+', line) is not None:
                return [s]