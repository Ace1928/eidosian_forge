from pprint import pformat
from six import iteritems
import re
@container_port.setter
def container_port(self, container_port):
    """
        Sets the container_port of this V1ContainerPort.
        Number of port to expose on the pod's IP address. This must be a valid
        port number, 0 < x < 65536.

        :param container_port: The container_port of this V1ContainerPort.
        :type: int
        """
    if container_port is None:
        raise ValueError('Invalid value for `container_port`, must not be `None`')
    self._container_port = container_port