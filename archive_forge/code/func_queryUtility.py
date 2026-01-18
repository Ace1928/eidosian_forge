from zope.interface.declarations import implementer
from zope.interface.interface import Attribute
from zope.interface.interface import Interface
def queryUtility(interface, name='', default=None):
    """Look up a utility that provides an interface.

        If one is not found, returns default.
        """