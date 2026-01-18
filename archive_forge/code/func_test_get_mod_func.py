from sympy.utilities.source import get_mod_func, get_class
def test_get_mod_func():
    assert get_mod_func('sympy.core.basic.Basic') == ('sympy.core.basic', 'Basic')