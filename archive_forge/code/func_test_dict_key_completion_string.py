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
def test_dict_key_completion_string(self):
    """Test dictionary key completion for string keys"""
    ip = get_ipython()
    complete = ip.Completer.complete
    ip.user_ns['d'] = {'abc': None}
    _, matches = complete(line_buffer='d[')
    self.assertIn("'abc'", matches)
    self.assertNotIn("'abc']", matches)
    _, matches = complete(line_buffer="d['")
    self.assertIn('abc', matches)
    self.assertNotIn("abc']", matches)
    _, matches = complete(line_buffer="d['a")
    self.assertIn('abc', matches)
    self.assertNotIn("abc']", matches)
    _, matches = complete(line_buffer='d["')
    self.assertIn('abc', matches)
    self.assertNotIn('abc"]', matches)
    _, matches = complete(line_buffer='d["a')
    self.assertIn('abc', matches)
    self.assertNotIn('abc"]', matches)
    _, matches = complete(line_buffer='d[]', cursor_pos=2)
    self.assertIn("'abc'", matches)
    _, matches = complete(line_buffer="d['']", cursor_pos=3)
    self.assertIn('abc', matches)
    self.assertNotIn("abc'", matches)
    self.assertNotIn("abc']", matches)
    ip.user_ns['d'] = {'abc': None, 'abd': None, 'bad': None, object(): None, 5: None, ('abe', None): None, (None, 'abf'): None}
    _, matches = complete(line_buffer="d['a")
    self.assertIn('abc', matches)
    self.assertIn('abd', matches)
    self.assertNotIn('bad', matches)
    self.assertNotIn('abe', matches)
    self.assertNotIn('abf', matches)
    assert not any((m.endswith((']', '"', "'")) for m in matches)), matches
    ip.user_ns['d'] = {'a\nb': None, "a'b": None, 'a"b': None, 'a word': None}
    _, matches = complete(line_buffer="d['a")
    self.assertIn('a\\nb', matches)
    self.assertIn("a\\'b", matches)
    self.assertIn('a"b', matches)
    self.assertIn('a word', matches)
    assert not any((m.endswith((']', '"', "'")) for m in matches)), matches
    _, matches = complete(line_buffer="d['a w")
    self.assertIn('word', matches)
    _, matches = complete(line_buffer="d['a\\'")
    self.assertIn('b', matches)
    _, matches = complete(line_buffer='d[')
    self.assertIn('"a\'b"', matches)
    _, matches = complete(line_buffer='d["a\'')
    self.assertIn('b', matches)
    if '-' not in ip.Completer.splitter.delims:
        ip.user_ns['d'] = {'before-after': None}
        _, matches = complete(line_buffer="d['before-af")
        self.assertIn('before-after', matches)
    ip.user_ns['d'] = {('foo', 'bar'): None}
    _, matches = complete(line_buffer='d[')
    self.assertIn("'foo'", matches)
    self.assertNotIn("'foo']", matches)
    self.assertNotIn("'bar'", matches)
    self.assertNotIn('foo', matches)
    self.assertNotIn('bar', matches)
    _, matches = complete(line_buffer="d['f")
    self.assertIn('foo', matches)
    self.assertNotIn("foo']", matches)
    self.assertNotIn('foo"]', matches)
    _, matches = complete(line_buffer="d['foo")
    self.assertIn('foo', matches)
    _, matches = complete(line_buffer="d['foo', ")
    self.assertIn("'bar'", matches)
    _, matches = complete(line_buffer="d['foo', 'b")
    self.assertIn('bar', matches)
    self.assertNotIn('foo', matches)
    _, matches = complete(line_buffer="d['foo', 'f")
    self.assertNotIn('bar', matches)
    self.assertNotIn('foo', matches)
    _, matches = complete(line_buffer="d['foo',]", cursor_pos=8)
    self.assertIn("'bar'", matches)
    self.assertNotIn('bar', matches)
    self.assertNotIn("'foo'", matches)
    self.assertNotIn('foo', matches)
    _, matches = complete(line_buffer="d['']", cursor_pos=3)
    self.assertIn('foo', matches)
    assert not any((m.endswith((']', '"', "'")) for m in matches)), matches
    _, matches = complete(line_buffer='d[""]', cursor_pos=3)
    self.assertIn('foo', matches)
    assert not any((m.endswith((']', '"', "'")) for m in matches)), matches
    _, matches = complete(line_buffer='d["foo","]', cursor_pos=9)
    self.assertIn('bar', matches)
    assert not any((m.endswith((']', '"', "'")) for m in matches)), matches
    _, matches = complete(line_buffer='d["foo",]', cursor_pos=8)
    self.assertIn("'bar'", matches)
    self.assertNotIn('bar', matches)
    ip.user_ns['d'] = {('foo', 'bar', 'foobar'): None}
    _, matches = complete(line_buffer="d['foo', 'b")
    self.assertIn('bar', matches)
    self.assertNotIn('foo', matches)
    self.assertNotIn('foobar', matches)
    _, matches = complete(line_buffer="d['foo', 'bar', 'fo")
    self.assertIn('foobar', matches)
    self.assertNotIn('foo', matches)
    self.assertNotIn('bar', matches)