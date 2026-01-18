from zope.interface.declarations import implementer
from zope.interface.interface import Attribute
from zope.interface.interface import Interface
def getMultiAdapter(objects, interface, name=''):
    """Look for a multi-adapter to an interface for multiple objects

        If a matching adapter cannot be found, a `ComponentLookupError`
        is raised.
        """