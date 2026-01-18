from io import BytesIO
from .. import errors, filters
from ..filters import (ContentFilter, ContentFilterContext,
from ..osutils import sha_string
from . import TestCase, TestCaseInTempDir
def test_filter_stack_maps(self):
    original_registry = filters._reset_registry()
    self.addCleanup(filters._reset_registry, original_registry)
    a_stack = [ContentFilter('b', 'c')]
    z_stack = [ContentFilter('y', 'x'), ContentFilter('w', 'v')]
    self._register_map('foo', a_stack, z_stack)
    self.assertEqual(['foo'], _get_registered_names())
    self._register_map('bar', z_stack, a_stack)
    self.assertEqual(['bar', 'foo'], _get_registered_names())
    self.assertRaises(KeyError, self._register_map, 'foo', [], [])