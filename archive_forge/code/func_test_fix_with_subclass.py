import numpy as np
import numpy.core as nx
import numpy.lib.ufunclike as ufl
from numpy.testing import (
def test_fix_with_subclass(self):

    class MyArray(nx.ndarray):

        def __new__(cls, data, metadata=None):
            res = nx.array(data, copy=True).view(cls)
            res.metadata = metadata
            return res

        def __array_wrap__(self, obj, context=None):
            if isinstance(obj, MyArray):
                obj.metadata = self.metadata
            return obj

        def __array_finalize__(self, obj):
            self.metadata = getattr(obj, 'metadata', None)
            return self
    a = nx.array([1.1, -1.1])
    m = MyArray(a, metadata='foo')
    f = ufl.fix(m)
    assert_array_equal(f, nx.array([1, -1]))
    assert_(isinstance(f, MyArray))
    assert_equal(f.metadata, 'foo')
    m0d = m[0, ...]
    m0d.metadata = 'bar'
    f0d = ufl.fix(m0d)
    assert_(isinstance(f0d, MyArray))
    assert_equal(f0d.metadata, 'bar')