import inspect
import locale
import os
import pydoc
import re
import textwrap
import warnings
from collections import defaultdict, deque
from itertools import chain, combinations, islice, tee
from pprint import pprint
from urllib.request import (
from nltk.collections import *
from nltk.internals import deprecated, raise_unorderable_types, slice_bounds
def everygrams(sequence, min_len=1, max_len=-1, pad_left=False, pad_right=False, **kwargs):
    """
    Returns all possible ngrams generated from a sequence of items, as an iterator.

        >>> sent = 'a b c'.split()

    New version outputs for everygrams.
        >>> list(everygrams(sent))
        [('a',), ('a', 'b'), ('a', 'b', 'c'), ('b',), ('b', 'c'), ('c',)]

    Old version outputs for everygrams.
        >>> sorted(everygrams(sent), key=len)
        [('a',), ('b',), ('c',), ('a', 'b'), ('b', 'c'), ('a', 'b', 'c')]

        >>> list(everygrams(sent, max_len=2))
        [('a',), ('a', 'b'), ('b',), ('b', 'c'), ('c',)]

    :param sequence: the source data to be converted into ngrams. If max_len is
        not provided, this sequence will be loaded into memory
    :type sequence: sequence or iter
    :param min_len: minimum length of the ngrams, aka. n-gram order/degree of ngram
    :type  min_len: int
    :param max_len: maximum length of the ngrams (set to length of sequence by default)
    :type  max_len: int
    :param pad_left: whether the ngrams should be left-padded
    :type pad_left: bool
    :param pad_right: whether the ngrams should be right-padded
    :type pad_right: bool
    :rtype: iter(tuple)
    """
    if max_len == -1:
        try:
            max_len = len(sequence)
        except TypeError:
            sequence = list(sequence)
            max_len = len(sequence)
    sequence = pad_sequence(sequence, max_len, pad_left, pad_right, **kwargs)
    history = list(islice(sequence, max_len))
    while history:
        for ngram_len in range(min_len, len(history) + 1):
            yield tuple(history[:ngram_len])
        try:
            history.append(next(sequence))
        except StopIteration:
            pass
        del history[0]