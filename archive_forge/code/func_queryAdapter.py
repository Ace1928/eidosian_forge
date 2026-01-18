from zope.interface.declarations import implementer
from zope.interface.interface import Attribute
from zope.interface.interface import Interface
def queryAdapter(object, interface, name='', default=None):
    """Look for a named adapter to an interface for an object

        If a matching adapter cannot be found, returns the default.
        """