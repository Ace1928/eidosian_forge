import re
import copy
import time
import base64
import random
import collections
from xml.dom import minidom
from datetime import datetime
from xml.sax.saxutils import escape as xml_escape
from libcloud.utils.py3 import ET, httplib, urlparse
from libcloud.utils.py3 import urlquote as url_quote
from libcloud.utils.py3 import _real_unicode, ensure_string
from libcloud.utils.misc import ReprMixin
from libcloud.common.azure import AzureRedirectException, AzureServiceManagementConnection
from libcloud.common.types import LibcloudError
from libcloud.compute.base import (
from libcloud.compute.types import NodeState
from libcloud.compute.providers import Provider
def ex_set_instance_endpoints(self, node, endpoints, ex_deployment_slot='Production'):
    """
        For example::

            endpoint = ConfigurationSetInputEndpoint(
                name='SSH',
                protocol='tcp',
                port=port,
                local_port='22',
                load_balanced_endpoint_set_name=None,
                enable_direct_server_return=False
            )
            {
                'name': 'SSH',
                'protocol': 'tcp',
                'port': port,
                'local_port': '22'
            }
        """
    ex_cloud_service_name = node.extra['ex_cloud_service_name']
    vm_role_name = node.name
    network_config = ConfigurationSet()
    network_config.configuration_set_type = 'NetworkConfiguration'
    for endpoint in endpoints:
        new_endpoint = ConfigurationSetInputEndpoint(**endpoint)
        network_config.input_endpoints.items.append(new_endpoint)
    _deployment_name = self._get_deployment(service_name=ex_cloud_service_name, deployment_slot=ex_deployment_slot).name
    response = self._perform_put(self._get_role_path(ex_cloud_service_name, _deployment_name, vm_role_name), AzureXmlSerializer.add_role_to_xml(None, None, None, 'PersistentVMRole', network_config, None, None, None, None))
    self.raise_for_response(response, 202)