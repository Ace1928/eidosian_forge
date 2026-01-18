from io import BytesIO
from .. import errors, filters
from ..filters import (ContentFilter, ContentFilterContext,
from ..osutils import sha_string
from . import TestCase, TestCaseInTempDir
def test_get_filter_stack_for(self):
    original_registry = filters._reset_registry()
    self.addCleanup(filters._reset_registry, original_registry)
    a_stack = [ContentFilter('b', 'c')]
    d_stack = [ContentFilter('d', 'D')]
    z_stack = [ContentFilter('y', 'x'), ContentFilter('w', 'v')]
    self._register_map('foo', a_stack, z_stack)
    self._register_map('bar', d_stack, z_stack)
    prefs = (('foo', 'v1'),)
    self.assertEqual(a_stack, _get_filter_stack_for(prefs))
    prefs = (('foo', 'v2'),)
    self.assertEqual(z_stack, _get_filter_stack_for(prefs))
    prefs = (('foo', 'v1'), ('bar', 'v1'))
    self.assertEqual(a_stack + d_stack, _get_filter_stack_for(prefs))
    prefs = (('baz', 'v1'),)
    self.assertEqual([], _get_filter_stack_for(prefs))
    prefs = (('foo', 'v3'),)
    self.assertEqual([], _get_filter_stack_for(prefs))
    prefs = (('foo', None), ('bar', 'v1'))
    self.assertEqual(d_stack, _get_filter_stack_for(prefs))