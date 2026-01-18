import abc
from neutron_lib.api.definitions import portbindings
@abc.abstractmethod
def is_partial_segment(self, segment):
    """Return True if segment is a partially specified segment.

        :param segment: segment dictionary
        :returns: boolean
        """