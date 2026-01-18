from ast import parse
import codecs
import collections
import operator
import os
import re
import timeit
from .compat import importlib_metadata_get
def read_python_file(path):
    fp = open(path, 'rb')
    try:
        encoding = parse_encoding(fp)
        data = fp.read()
        if encoding:
            data = data.decode(encoding)
        return data
    finally:
        fp.close()