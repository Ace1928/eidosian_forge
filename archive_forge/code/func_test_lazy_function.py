import sys
from sympy.core.cache import cacheit, cached_property, lazy_function
from sympy.testing.pytest import raises
def test_lazy_function():
    module_name = 'xmlrpc.client'
    function_name = 'gzip_decode'
    lazy = lazy_function(module_name, function_name)
    assert lazy(b'') == b''
    assert module_name in sys.modules
    assert function_name in str(lazy)
    repr_lazy = repr(lazy)
    assert 'LazyFunction' in repr_lazy
    assert function_name in repr_lazy
    lazy = lazy_function('sympy.core.cache', 'cheap')