from typing import cast
from zope.interface import Attribute, Interface, implementer
from twisted.web import sux
def compareAttribute(self, attrib, value):
    """Safely compare the value of an attribute against a provided value.

        L{None}-safe.
        """
    return self.attributes.get(self._dqa(attrib), None) == value