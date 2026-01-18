from .api.client import APIClient
from .constants import (DEFAULT_TIMEOUT_SECONDS, DEFAULT_MAX_POOL_SIZE)
from .models.configs import ConfigCollection
from .models.containers import ContainerCollection
from .models.images import ImageCollection
from .models.networks import NetworkCollection
from .models.nodes import NodeCollection
from .models.plugins import PluginCollection
from .models.secrets import SecretCollection
from .models.services import ServiceCollection
from .models.swarm import Swarm
from .models.volumes import VolumeCollection
from .utils import kwargs_from_env
from_env = DockerClient.from_env
@property
def configs(self):
    """
        An object for managing configs on the server. See the
        :doc:`configs documentation <configs>` for full details.
        """
    return ConfigCollection(client=self)