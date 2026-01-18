from ..errors import InvalidVersion
from ..utils import check_resource, minimum_version
from ..utils import version_lt
from .. import utils
@check_resource('net_id')
def remove_network(self, net_id):
    """
        Remove a network. Similar to the ``docker network rm`` command.

        Args:
            net_id (str): The network's id
        """
    url = self._url('/networks/{0}', net_id)
    res = self._delete(url)
    self._raise_for_status(res)