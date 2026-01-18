from zope.interface.declarations import implementer
from zope.interface.interface import Attribute
from zope.interface.interface import Interface
def registeredUtilities():
    """Return an iterable of `IUtilityRegistration` instances.

        These registrations describe the current utility registrations
        in the object.
        """