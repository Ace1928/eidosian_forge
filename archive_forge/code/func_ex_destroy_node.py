import json
import hashlib
import datetime
from typing import Any, Dict, List, Union, Optional
from collections import OrderedDict
from libcloud.compute.base import Node, NodeSize, NodeImage
from libcloud.compute.types import NodeState
from libcloud.container.base import Container, ContainerImage, ContainerDriver, ContainerCluster
from libcloud.container.types import ContainerState
from libcloud.common.exceptions import BaseHTTPError
from libcloud.common.kubernetes import (
from libcloud.container.providers import Provider
def ex_destroy_node(self, node_name: str) -> bool:
    """
        Destroy a node.

        :param node_name: Name of the node to destroy
        :type  node_name: ``str``

        :rtype: ``bool``
        """
    self.connection.request(ROOT_URL + f'v1/nodes/{node_name}', method='DELETE').object
    return True