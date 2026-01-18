from datetime import datetime
from .. import errors
from .. import utils
from ..constants import DEFAULT_DATA_CHUNK_SIZE
from ..types import CancellableStream
from ..types import ContainerConfig
from ..types import EndpointConfig
from ..types import HostConfig
from ..types import NetworkingConfig
def create_container_from_config(self, config, name=None, platform=None):
    u = self._url('/containers/create')
    params = {'name': name}
    if platform:
        if utils.version_lt(self._version, '1.41'):
            raise errors.InvalidVersion('platform is not supported for API version < 1.41')
        params['platform'] = platform
    res = self._post_json(u, data=config, params=params)
    return self._result(res, True)