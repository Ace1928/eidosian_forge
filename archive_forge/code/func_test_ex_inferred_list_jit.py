import unittest
from numba.tests.support import captured_stdout
def test_ex_inferred_list_jit(self):
    with captured_stdout():
        from numba import njit
        from numba.typed import List

        @njit
        def foo():
            l = List()
            l.append(42)
            print(l[0])
            l[0] = 23
            print(l[0])
            print(len(l))
            l.pop()
            print(len(l))
            return l
        foo()