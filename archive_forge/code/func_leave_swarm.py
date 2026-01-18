import logging
import http.client as http_client
from ..constants import DEFAULT_SWARM_ADDR_POOL, DEFAULT_SWARM_SUBNET_SIZE
from .. import errors
from .. import types
from .. import utils
@utils.minimum_version('1.24')
def leave_swarm(self, force=False):
    """
        Leave a swarm.

        Args:
            force (bool): Leave the swarm even if this node is a manager.
                Default: ``False``

        Returns:
            ``True`` if the request went through.

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.
        """
    url = self._url('/swarm/leave')
    response = self._post(url, params={'force': force})
    if force and response.status_code == http_client.NOT_ACCEPTABLE:
        return True
    if force and response.status_code == http_client.SERVICE_UNAVAILABLE:
        return True
    self._raise_for_status(response)
    return True