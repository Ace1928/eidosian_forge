from .. import errors
from ..filters import _get_filter_stack_for
from ..filters.eol import _to_crlf_converter, _to_lf_converter
from . import TestCase
def test_exact_value(self):
    """'eol = exact' should have no content filters"""
    prefs = (('eol', 'exact'),)
    self.assertEqual([], _get_filter_stack_for(prefs))