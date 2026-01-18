from zope.interface.declarations import implementer
from zope.interface.interface import Attribute
from zope.interface.interface import Interface
def registeredHandlers():
    """Return an iterable of `IHandlerRegistration` instances.

        These registrations describe the current handler registrations
        in the object.
        """