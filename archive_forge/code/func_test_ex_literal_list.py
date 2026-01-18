import unittest
from numba.tests.support import captured_stdout
from numba import typed
def test_ex_literal_list(self):
    with captured_stdout():
        from numba import njit
        from numba.extending import overload

        def specialize(x):
            pass

        @overload(specialize)
        def ol_specialize(x):
            l = x.literal_value
            const_expr = []
            for v in l:
                const_expr.append(str(v))
            const_strings = tuple(const_expr)

            def impl(x):
                return const_strings
            return impl

        @njit
        def foo():
            const_list = ['a', 10, 1j, ['another', 'list']]
            return specialize(const_list)
        result = foo()
        print(result)
    expected = ('Literal[str](a)', 'Literal[int](10)', 'complex128', "list(unicode_type)<iv=['another', 'list']>")
    self.assertEqual(result, expected)