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
def put_archive(self, container, path, data):
    """
        Insert a file or folder in an existing container using a tar archive as
        source.

        Args:
            container (str): The container where the file(s) will be extracted
            path (str): Path inside the container where the file(s) will be
                extracted. Must exist.
            data (bytes or stream): tar data to be extracted

        Returns:
            (bool): True if the call succeeds.

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.
        """
    params = {'path': path}
    url = self._url('/containers/{0}/archive', container)
    res = self._put(url, params=params, data=data)
    self._raise_for_status(res)
    return res.status_code == 200