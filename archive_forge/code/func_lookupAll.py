from zope.interface.declarations import implementer
from zope.interface.interface import Attribute
from zope.interface.interface import Interface
def lookupAll(required, provided):
    """Find all adapters from the required to the provided interfaces

        An iterable object is returned that provides name-value two-tuples.
        """