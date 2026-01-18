import operator
import sys
import types
import unittest
import abc
import pytest
import six
def test_import_moves_error_2():
    from six import moves
    assert moves.urllib.parse.urljoin
    from six.moves.urllib.parse import urljoin