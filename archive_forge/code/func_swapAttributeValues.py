from typing import cast
from zope.interface import Attribute, Interface, implementer
from twisted.web import sux
def swapAttributeValues(self, left, right):
    """Swap the values of two attribute."""
    d = self.attributes
    l = d[left]
    d[left] = d[right]
    d[right] = l