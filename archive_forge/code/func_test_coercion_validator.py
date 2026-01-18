import pytest
from traitlets import HasTraits, TraitError
from ..traittypes import SciType
def test_coercion_validator():

    def truncate(trait, value):
        return value[:10]

    class Foo(HasTraits):
        bar = SciType().valid(truncate)
    foo = Foo(bar=list(range(20)))
    assert foo.bar == list(range(10))
    foo.bar = list(range(10, 40))
    assert foo.bar == list(range(10, 20))