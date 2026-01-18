import pytest
from nibabel.onetime import auto_attr, setattr_on_read
from nibabel.testing import deprecated_to, expires
def test_auto_attr():

    class MagicProp:

        @auto_attr
        def a(self):
            return object()
    x = MagicProp()
    assert 'a' not in x.__dict__
    obj = x.a
    assert 'a' in x.__dict__
    assert x.a is obj