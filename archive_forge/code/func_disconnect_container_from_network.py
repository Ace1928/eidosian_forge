from ..errors import InvalidVersion
from ..utils import check_resource, minimum_version
from ..utils import version_lt
from .. import utils
@check_resource('container')
def disconnect_container_from_network(self, container, net_id, force=False):
    """
        Disconnect a container from a network.

        Args:
            container (str): container ID or name to be disconnected from the
                network
            net_id (str): network ID
            force (bool): Force the container to disconnect from a network.
                Default: ``False``
        """
    data = {'Container': container}
    if force:
        if version_lt(self._version, '1.22'):
            raise InvalidVersion('Forced disconnect was introduced in API 1.22')
        data['Force'] = force
    url = self._url('/networks/{0}/disconnect', net_id)
    res = self._post_json(url, data=data)
    self._raise_for_status(res)