from abc import ABC
from ..coretypes import (
from ..dispatch import dispatch
@dispatch(Function, Function)
def assert_dshape_equal(a, b, path=None, **kwargs):
    assert len(a.argtypes) == len(b.argtypes), 'functions have different arities: %d != %d\n%r != %r\n%s' % (len(a.argtypes), len(b.argtypes), a, b, _fmt_path(path))
    if path is None:
        path = ()
    for n, (aarg, barg) in enumerate(zip(a.argtypes, b.argtypes)):
        assert_dshape_equal(aarg, barg, path=path + ('.argtypes[%d]' % n,), **kwargs)
    assert_dshape_equal(a.restype, b.restype, path=path + ('.restype',), **kwargs)