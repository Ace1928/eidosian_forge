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
def test_function_pretty():
    """Test pretty print of function"""
    import posixpath
    assert pretty.pretty(posixpath.join) == '<function posixpath.join(a, *p)>'

    def meaning_of_life(question=None):
        if question:
            return 42
        return "Don't panic"
    assert 'meaning_of_life(question=None)' in pretty.pretty(meaning_of_life)