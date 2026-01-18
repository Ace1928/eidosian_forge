from sympy.external import import_module
from sympy.testing.pytest import ignore_warnings, raises
def test_no_import():
    from sympy.parsing.latex import parse_latex
    with ignore_warnings(UserWarning):
        with raises(ImportError):
            parse_latex('1 + 1')