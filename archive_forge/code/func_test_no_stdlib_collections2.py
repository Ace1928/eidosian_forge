from sympy.external import import_module
from sympy.testing.pytest import warns
def test_no_stdlib_collections2():
    """
    make sure we get the right collections when it is not part of a
    larger list
    """
    import collections
    matplotlib = import_module('matplotlib', import_kwargs={'fromlist': ['collections']}, min_module_version='1.1.0', catch=(RuntimeError,))
    if matplotlib:
        assert collections != matplotlib.collections