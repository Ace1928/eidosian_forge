import re
from .. import errors, lazy_regex
from ..globbing import (ExceptionGlobster, Globster, _OrderedGlobster,
from . import TestCase
def test_end_anchor(self):
    self.assertMatch([('*.333', ['foo.333'], ['foo.3']), ('*.3', ['foo.3'], ['foo.333'])])