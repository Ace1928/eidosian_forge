import abc
from neutron_lib.api.definitions import portbindings
@abc.abstractmethod
def allocate_dynamic_segment(self, segment):
    """Allocate a dynamic segment.

        :param segment: A partially or fully specified segment dictionary

        Called by the MechanismDriver.bind_port, create_port or update_port
        to dynamically allocate a segment for the port using the partial
        segment specified. The segment dictionary can be a fully or partially
        specified segment. At a minimum it needs the network_type populated to
        call on the appropriate type driver.
        """