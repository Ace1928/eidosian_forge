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
def binary_search_file(file, key, cache=None, cacheDepth=-1):
    """
    Return the line from the file with first word key.
    Searches through a sorted file using the binary search algorithm.

    :type file: file
    :param file: the file to be searched through.
    :type key: str
    :param key: the identifier we are searching for.
    """
    key = key + ' '
    keylen = len(key)
    start = 0
    currentDepth = 0
    if hasattr(file, 'name'):
        end = os.stat(file.name).st_size - 1
    else:
        file.seek(0, 2)
        end = file.tell() - 1
        file.seek(0)
    if cache is None:
        cache = {}
    while start < end:
        lastState = (start, end)
        middle = (start + end) // 2
        if cache.get(middle):
            offset, line = cache[middle]
        else:
            line = ''
            while True:
                file.seek(max(0, middle - 1))
                if middle > 0:
                    file.discard_line()
                offset = file.tell()
                line = file.readline()
                if line != '':
                    break
                middle = (start + middle) // 2
                if middle == end - 1:
                    return None
            if currentDepth < cacheDepth:
                cache[middle] = (offset, line)
        if offset > end:
            assert end != middle - 1, 'infinite loop'
            end = middle - 1
        elif line[:keylen] == key:
            return line
        elif line > key:
            assert end != middle - 1, 'infinite loop'
            end = middle - 1
        elif line < key:
            start = offset + len(line) - 1
        currentDepth += 1
        thisState = (start, end)
        if lastState == thisState:
            return None
    return None