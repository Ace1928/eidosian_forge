from __future__ import absolute_import
import pytest
from ..symbolic import SymbolicSys
from ..util import requires, import_
from .test_symbolic import decay_dydt_factory
@requires('numpy')
def test_import_():
    sqrt, sin = import_('numpy', 'sqrt', 'sin')
    assert sqrt(4) == 2 and sin(0) == 0
    foo, bar = import_('numpy', 'foo', 'bar')
    with pytest.raises(AttributeError):
        foo.baz
    with pytest.raises(AttributeError):
        bar(3)
    qux = import_('qux')
    with pytest.raises(ImportError):
        qux.__name__