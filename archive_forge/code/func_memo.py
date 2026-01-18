from ast import parse
import codecs
import collections
import operator
import os
import re
import timeit
from .compat import importlib_metadata_get
def memo(*a, **kw):
    return result