from __future__ import absolute_import
import ast
import textwrap
from ...TestUtils import CythonTest
from .. import ExprNodes
from ..Errors import CompileError
def test_invalid_ellipsis(self):
    ERR = ':{0}:{1}: Expected an identifier or literal'
    for code, line, col in INVALID_ELLIPSIS:
        try:
            ast.parse(textwrap.dedent(code))
        except SyntaxError as exc:
            assert True
        else:
            assert False, "Invalid Python code '%s' failed to raise an exception" % code
        try:
            self.fragment(u'                # cython: language_level=3\n                ' + code)
        except CompileError as exc:
            assert ERR.format(line, col) in str(exc), str(exc)
        else:
            assert False, "Invalid Cython code '%s' failed to raise an exception" % code