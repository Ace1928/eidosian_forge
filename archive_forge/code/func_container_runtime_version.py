from pprint import pformat
from six import iteritems
import re
@container_runtime_version.setter
def container_runtime_version(self, container_runtime_version):
    """
        Sets the container_runtime_version of this V1NodeSystemInfo.
        ContainerRuntime Version reported by the node through runtime remote API
        (e.g. docker://1.5.0).

        :param container_runtime_version: The container_runtime_version of this
        V1NodeSystemInfo.
        :type: str
        """
    if container_runtime_version is None:
        raise ValueError('Invalid value for `container_runtime_version`, must not be `None`')
    self._container_runtime_version = container_runtime_version