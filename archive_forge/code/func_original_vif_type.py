import abc
from neutron_lib.api.definitions import portbindings
@property
@abc.abstractmethod
def original_vif_type(self):
    """Return the original vif_type of the port.

        In the context of a host-specific operation on a distributed
        port, the original_vif_type property indicates original
        binding state for the host for which the port operation is
        being performed. Otherwise, it is the same value as
        original['binding:vif_type'].

        This property is only valid within calls to
        update_port_precommit and update_port_postcommit. It returns
        None otherwise.
        """