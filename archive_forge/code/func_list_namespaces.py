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
def list_namespaces(self) -> List[KubernetesNamespace]:
    """
        Get a list of namespaces that pods can be deployed into

        :rtype: ``list`` of :class:`.KubernetesNamespace`
        """
    try:
        result = self.connection.request(ROOT_URL + 'v1/namespaces/').object
    except Exception as exc:
        errno = getattr(exc, 'errno', None)
        if errno == 111:
            raise KubernetesException(errno, 'Make sure kube host is accessibleand the API port is correct')
        raise
    namespaces = [self._to_namespace(value) for value in result['items']]
    return namespaces