import warnings
from bs4.element import (
from . import SoupTest
def test_has_attr(self):
    """has_attr() checks for the presence of an attribute.

        Please note note: has_attr() is different from
        __in__. has_attr() checks the tag's attributes and __in__
        checks the tag's chidlren.
        """
    soup = self.soup("<foo attr='bar'>")
    assert soup.foo.has_attr('attr')
    assert not soup.foo.has_attr('attr2')