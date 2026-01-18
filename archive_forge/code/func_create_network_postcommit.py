import abc
from neutron_lib.api.definitions import portbindings
def create_network_postcommit(self, context):
    """Create a network.

        :param context: NetworkContext instance describing the new
            network.

        Called after the transaction commits. Call can block, though
        will block the entire process so care should be taken to not
        drastically affect performance. Raising an exception will
        cause the deletion of the resource.
        """
    pass