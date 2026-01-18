from zope.interface.declarations import implementer
from zope.interface.interface import Attribute
from zope.interface.interface import Interface
def registeredSubscriptionAdapters():
    """Return an iterable of `ISubscriptionAdapterRegistration` instances.

        These registrations describe the current subscription adapter
        registrations in the object.
        """