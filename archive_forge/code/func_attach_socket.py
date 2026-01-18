from datetime import datetime
from .. import errors
from .. import utils
from ..constants import DEFAULT_DATA_CHUNK_SIZE
from ..types import CancellableStream
from ..types import ContainerConfig
from ..types import EndpointConfig
from ..types import HostConfig
from ..types import NetworkingConfig
@utils.check_resource('container')
def attach_socket(self, container, params=None, ws=False):
    """
        Like ``attach``, but returns the underlying socket-like object for the
        HTTP request.

        Args:
            container (str): The container to attach to.
            params (dict): Dictionary of request parameters (e.g. ``stdout``,
                ``stderr``, ``stream``).
                For ``detachKeys``, ~/.docker/config.json is used by default.
            ws (bool): Use websockets instead of raw HTTP.

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.
        """
    if params is None:
        params = {'stdout': 1, 'stderr': 1, 'stream': 1}
    if 'detachKeys' not in params and 'detachKeys' in self._general_configs:
        params['detachKeys'] = self._general_configs['detachKeys']
    if ws:
        return self._attach_websocket(container, params)
    headers = {'Connection': 'Upgrade', 'Upgrade': 'tcp'}
    u = self._url('/containers/{0}/attach', container)
    return self._get_raw_response_socket(self.post(u, None, params=self._attach_params(params), stream=True, headers=headers))