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
def test_collections_ordereddict():
    a = OrderedDict()
    a['key'] = a
    cases = [(OrderedDict(), 'OrderedDict()'), (OrderedDict(((i, i) for i in range(1000, 1010))), 'OrderedDict([(1000, 1000),\n             (1001, 1001),\n             (1002, 1002),\n             (1003, 1003),\n             (1004, 1004),\n             (1005, 1005),\n             (1006, 1006),\n             (1007, 1007),\n             (1008, 1008),\n             (1009, 1009)])'), (a, "OrderedDict([('key', OrderedDict(...))])")]
    for obj, expected in cases:
        assert pretty.pretty(obj) == expected