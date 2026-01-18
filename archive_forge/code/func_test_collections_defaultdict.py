from collections import Counter, defaultdict, deque, OrderedDict, UserList
import os
import pytest
import types
import string
import sys
import unittest
import pytest
from IPython.lib import pretty
from io import StringIO
def test_collections_defaultdict():
    a = defaultdict()
    a.default_factory = a
    b = defaultdict(list)
    b['key'] = b
    cases = [(defaultdict(list), 'defaultdict(list, {})'), (defaultdict(list, {'key': '-' * 50}), "defaultdict(list,\n            {'key': '--------------------------------------------------'})"), (a, 'defaultdict(defaultdict(...), {})'), (b, "defaultdict(list, {'key': defaultdict(...)})")]
    for obj, expected in cases:
        assert pretty.pretty(obj) == expected