from sympy.core.sorting import default_sort_key, ordered
from sympy.testing.pytest import raises
from sympy.abc import x
def test_default_sort_key():
    func = lambda x: x
    assert sorted([func, x, func], key=default_sort_key) == [func, func, x]

    class C:

        def __repr__(self):
            return 'x.y'
    func = C()
    assert sorted([x, func], key=default_sort_key) == [func, x]