import unittest
from numba.tests.support import captured_stdout
from numba import typed
def test_ex_initial_value_dict_compile_time_consts(self):
    with captured_stdout():
        from numba import njit, literally
        from numba.extending import overload

        def specialize(x):
            pass

        @overload(specialize)
        def ol_specialize(x):
            iv = x.initial_value
            if iv is None:
                return lambda x: literally(x)
            assert iv == {'a': 1, 'b': 2, 'c': 3}
            return lambda x: literally(x)

        @njit
        def foo():
            d = {'a': 1, 'b': 2, 'c': 3}
            d['c'] = 20
            d['d'] = 30
            return specialize(d)
        result = foo()
        print(result)
    expected = typed.Dict()
    for k, v in {'a': 1, 'b': 2, 'c': 20, 'd': 30}.items():
        expected[k] = v
    self.assertEqual(result, expected)