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
def test_bad_repr():
    """Don't catch bad repr errors"""
    with pytest.raises(ZeroDivisionError):
        pretty.pretty(BadRepr())