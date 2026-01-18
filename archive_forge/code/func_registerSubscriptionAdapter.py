from zope.interface.declarations import implementer
from zope.interface.interface import Attribute
from zope.interface.interface import Interface
def registerSubscriptionAdapter(factory, required=None, provides=None, name='', info=''):
    """Register a subscriber factory

        :param factory:
            The object used to compute the adapter

        :param required:
            This is a sequence of specifications for objects to be
            adapted.  If omitted, then the value of the factory's
            ``__component_adapts__`` attribute will be used.  The
            ``__component_adapts__`` attribute is
            normally set using the adapter
            decorator.  If the factory doesn't have a
            ``__component_adapts__`` adapts attribute, then this
            argument is required.

        :param provided:
            This is the interface provided by the adapter and
            implemented by the factory.  If the factory implements
            a single interface, then this argument is optional and
            the factory-implemented interface will be used.

        :param name:
            The adapter name.

            Currently, only the empty string is accepted.  Other
            strings will be accepted in the future when support for
            named subscribers is added.

        :param info:
           An object that can be converted to a string to provide
           information about the registration.

        A `IRegistered` event is generated with an
        `ISubscriptionAdapterRegistration`.
        """