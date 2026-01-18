from bs4.element import (
from . import SoupTest
def test_name_may_be_none_or_missing(self):
    a = NamespacedAttribute('xmlns', None)
    assert a == 'xmlns'
    a = NamespacedAttribute('xmlns', '')
    assert a == 'xmlns'
    a = NamespacedAttribute('xmlns')
    assert a == 'xmlns'