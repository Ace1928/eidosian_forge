import abc
from neutron_lib.api.definitions import portbindings
@property
@abc.abstractmethod
def original_status(self):
    """Return the status of the original port.

        The method is only valid within calls to update_port_precommit and
        update_port_postcommit.
        """