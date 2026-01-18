import re
import itertools
import os
import logging
import string
import pyparsing
from pyparsing import __version__ as pyparsing_version
from pyparsing import (Literal, CaselessLiteral, Word, OneOrMore, Forward, Group, Optional, Combine, restOfLine,
from collections import OrderedDict
def nsplit(seq, n=2):
    """Split a sequence into pieces of length n

    If the lengt of the sequence isn't a multiple of n, the rest is discareded.
    Note that nsplit will strings into individual characters.

    Examples:
    >>> nsplit('aabbcc')
    [('a', 'a'), ('b', 'b'), ('c', 'c')]
    >>> nsplit('aabbcc',n=3)
    [('a', 'a', 'b'), ('b', 'c', 'c')]

    # Note that cc is discarded
    >>> nsplit('aabbcc',n=4)
    [('a', 'a', 'b', 'b')]
    """
    return [xy for xy in zip(*[iter(seq)] * n)]