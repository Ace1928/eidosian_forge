import re
from .. import errors, lazy_regex
from ..globbing import (ExceptionGlobster, Globster, _OrderedGlobster,
from . import TestCase
def test_forward_slashes(self):
    """tests that multiple foward slashes are collapsed to single forward
        slashes and trailing forward slashes are removed"""
    self.assertEqual('/', normalize_pattern('/'))
    self.assertEqual('/', normalize_pattern('//'))
    self.assertEqual('/foo/bar', normalize_pattern('/foo/bar'))
    self.assertEqual('foo/bar', normalize_pattern('foo/bar/'))
    self.assertEqual('/foo/bar', normalize_pattern('//foo//bar//'))