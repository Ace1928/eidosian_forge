import os
import re
import copy
import time
import base64
import datetime
from xml.parsers.expat import ExpatError
from libcloud.utils.py3 import ET, b, next, httplib, urlparse, urlencode
from libcloud.common.base import XmlResponse, ConnectionUserAndKey
from libcloud.common.types import LibcloudError, InvalidCredsError
from libcloud.compute.base import Node, NodeSize, NodeImage, NodeDriver, NodeLocation
from libcloud.compute.types import NodeState
from libcloud.utils.iso8601 import parse_date
from libcloud.compute.providers import Provider
def ex_deploy_node(self, node, ex_force_customization=False):
    """
        Deploys existing node. Equal to vApp "start" operation.

        :param  node: The node to be deployed
        :type   node: :class:`Node`

        :param  ex_force_customization: Used to specify whether to force
                                        customization on deployment,
                                        if not set default value is False.
        :type   ex_force_customization: ``bool``

        :rtype: :class:`Node`
        """
    if ex_force_customization:
        vms = self._get_vm_elements(node.id)
        for vm in vms:
            self._ex_deploy_node_or_vm(vm.get('href'), ex_force_customization=True)
    else:
        self._ex_deploy_node_or_vm(node.id)
    res = self.connection.request(get_url_path(node.id))
    return self._to_node(res.object)