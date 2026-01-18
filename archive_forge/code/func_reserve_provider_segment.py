import abc
from neutron_lib.api.definitions import portbindings
@abc.abstractmethod
def reserve_provider_segment(self, context, segment, filters=None):
    """Reserve resource associated with a provider network segment.

        :param context: instance of neutron context with DB session
        :param segment: segment dictionary
        :param filters: a dictionary that is used as search criteria
        :returns: segment dictionary

        Called inside transaction context on session to reserve the
        type-specific resource for a provider network segment. The
        segment dictionary passed in was returned by a previous
        validate_provider_segment() call.
        """