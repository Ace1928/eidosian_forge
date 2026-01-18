from datetime import datetime
from libcloud.common.gandi import (
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
def deploy_node(self, **kwargs):
    """
        deploy_node is not implemented for gandi driver

        :rtype: ``bool``
        """
    raise NotImplementedError('deploy_node not implemented for gandi driver')