import operator
import sys
import types
import unittest
import abc
import pytest
import six
def test_lazy():
    if six.PY3:
        html_name = 'html.parser'
    else:
        html_name = 'HTMLParser'
    assert html_name not in sys.modules
    mod = six.moves.html_parser
    assert sys.modules[html_name] is mod
    assert 'htmlparser' not in six._MovedItems.__dict__