import abc
from neutron_lib.api.definitions import portbindings
@property
@abc.abstractmethod
def network_segments(self):
    """Return the segments associated with this network resource."""