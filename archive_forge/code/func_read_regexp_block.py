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
def read_regexp_block(stream, start_re, end_re=None):
    """
    Read a sequence of tokens from a stream, where tokens begin with
    lines that match ``start_re``.  If ``end_re`` is specified, then
    tokens end with lines that match ``end_re``; otherwise, tokens end
    whenever the next line matching ``start_re`` or EOF is found.
    """
    while True:
        line = stream.readline()
        if not line:
            return []
        if re.match(start_re, line):
            break
    lines = [line]
    while True:
        oldpos = stream.tell()
        line = stream.readline()
        if not line:
            return [''.join(lines)]
        if end_re is not None and re.match(end_re, line):
            return [''.join(lines)]
        if end_re is None and re.match(start_re, line):
            stream.seek(oldpos)
            return [''.join(lines)]
        lines.append(line)