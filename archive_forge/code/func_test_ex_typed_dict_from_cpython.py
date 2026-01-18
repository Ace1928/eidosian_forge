import unittest
from numba.tests.support import captured_stdout
def test_ex_typed_dict_from_cpython(self):
    with captured_stdout():
        import numpy as np
        from numba import njit
        from numba.core import types
        from numba.typed import Dict
        d = Dict.empty(key_type=types.unicode_type, value_type=types.float64[:])
        d['posx'] = np.asarray([1, 0.5, 2], dtype='f8')
        d['posy'] = np.asarray([1.5, 3.5, 2], dtype='f8')
        d['velx'] = np.asarray([0.5, 0, 0.7], dtype='f8')
        d['vely'] = np.asarray([0.2, -0.2, 0.1], dtype='f8')

        @njit
        def move(d):
            d['posx'] += d['velx']
            d['posy'] += d['vely']
        print('posx: ', d['posx'])
        print('posy: ', d['posy'])
        move(d)
        print('posx: ', d['posx'])
        print('posy: ', d['posy'])
    np.testing.assert_array_equal(d['posx'], [1.5, 0.5, 2.7])
    np.testing.assert_array_equal(d['posy'], [1.7, 3.3, 2.1])