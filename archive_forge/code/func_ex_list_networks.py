import os
import re
import base64
import collections
from libcloud.utils.py3 import b, httplib
from libcloud.common.base import JsonResponse, ConnectionUserAndKey, KeyCertificateConnection
from libcloud.common.types import InvalidCredsError
from libcloud.compute.base import StorageVolume
from libcloud.container.base import Container, ContainerImage, ContainerDriver
from libcloud.container.types import ContainerState
from libcloud.common.exceptions import BaseHTTPError
from libcloud.container.providers import Provider
def ex_list_networks(self):
    """
        Returns a list of networks.
        Implements GET /1.0/networks
        Authentication: trusted
        Operation: sync

        :rtype: list of LXDNetwork objects
        """
    req = '/%s/networks' % self.version
    response = self.connection.request(req)
    response_dict = response.parse_body()
    assert_response(response_dict=response_dict, status_code=200)
    nets = response_dict['metadata']
    networks = []
    for net in nets:
        name = net.split('/')[-1]
        networks.append(self.ex_get_network(name=name))
    return networks