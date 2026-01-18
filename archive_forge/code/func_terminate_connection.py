from cinderclient.apiclient import base as common_base
from cinderclient import base
def terminate_connection(self, volume, connector):
    """Terminate a volume connection.

        :param volume: The :class:`Volume` (or its ID).
        :param connector: connector dict from nova.
        """
    return self._action('os-terminate_connection', volume, {'connector': connector})