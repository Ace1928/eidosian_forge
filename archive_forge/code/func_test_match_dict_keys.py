import os
import pytest
import sys
import textwrap
import unittest
from contextlib import contextmanager
from traitlets.config.loader import Config
from IPython import get_ipython
from IPython.core import completer
from IPython.utils.tempdir import TemporaryDirectory, TemporaryWorkingDirectory
from IPython.utils.generics import complete_object
from IPython.testing import decorators as dec
from IPython.core.completer import (
def test_match_dict_keys(self):
    """
        Test that match_dict_keys works on a couple of use case does return what
        expected, and does not crash
        """
    delims = ' \t\n`!@#$^&*()=+[{]}\\|;:\'",<>?'

    def match(*args, **kwargs):
        quote, offset, matches = match_dict_keys(*args, delims=delims, **kwargs)
        return (quote, offset, list(matches))
    keys = ['foo', b'far']
    assert match(keys, "b'") == ("'", 2, ['far'])
    assert match(keys, "b'f") == ("'", 2, ['far'])
    assert match(keys, 'b"') == ('"', 2, ['far'])
    assert match(keys, 'b"f') == ('"', 2, ['far'])
    assert match(keys, "'") == ("'", 1, ['foo'])
    assert match(keys, "'f") == ("'", 1, ['foo'])
    assert match(keys, '"') == ('"', 1, ['foo'])
    assert match(keys, '"f') == ('"', 1, ['foo'])
    keys = [('foo', 1111), ('foo', 2222), (3333, 'bar'), (3333, 'test')]
    assert match(keys, "'f") == ("'", 1, ['foo'])
    assert match(keys, '33') == ('', 0, ['3333'])
    keys = [3735928559, 1111, 1234, '1999', 21, 22]
    assert match(keys, '0xdead') == ('', 0, ['0xdeadbeef'])
    assert match(keys, '1') == ('', 0, ['1111', '1234'])
    assert match(keys, '2') == ('', 0, ['21', '22'])
    assert match(keys, '0b101') == ('', 0, ['0b10101', '0b10110'])
    assert match(keys, 'a_variable') == ('', 0, [])
    assert match(keys, "'' ''") == ('', 0, [])