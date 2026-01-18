from __future__ import absolute_import
import ast
import textwrap
from ...TestUtils import CythonTest
from .. import ExprNodes
from ..Errors import CompileError
def test_invalid_number_literals(self):
    for literal in INVALID_UNDERSCORE_LITERALS:
        for expression in ['%s', '1 + %s', '%s + 1', '2 * %s', '%s * 2']:
            code = 'x = ' + expression % literal
            try:
                self.fragment(u'                    # cython: language_level=3\n                    ' + code)
            except CompileError as exc:
                assert code in [s.strip() for s in str(exc).splitlines()], str(exc)
            else:
                assert False, "Invalid Cython code '%s' failed to raise an exception" % code