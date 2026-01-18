import abc
from neutron_lib.api.definitions import portbindings
def create_port_postcommit(self, context):
    """Create a port.

        :param context: PortContext instance describing the port.

        Called after the transaction completes. Call can block, though
        will block the entire process so care should be taken to not
        drastically affect performance.  Raising an exception will
        result in the deletion of the resource.
        """
    pass