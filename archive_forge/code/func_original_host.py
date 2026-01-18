import abc
from neutron_lib.api.definitions import portbindings
@property
@abc.abstractmethod
def original_host(self):
    """Return the original host with which the port was associated.

        In the context of a host-specific operation on a distributed
        port, the original_host property indicates the host for which
        the port operation is being performed. Otherwise, it is the
        same value as original['binding:host_id'].

        This property is only valid within calls to
        update_port_precommit and update_port_postcommit. It returns
        None otherwise.
        """