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
def ex_create_network(self, name, **kwargs):
    """
        Create a new network with the given name and
        and the specified configuration

        Authentication: trusted
        Operation: sync

        :param name: The name of the new network
        :type  name: str
        """
    kwargs['name'] = name
    data = json.dumps(kwargs)
    req = '/%s/networks' % self.version
    response = self.connection.request(req, method='POST', data=data)
    response_dict = response.parse_body()
    assert_response(response_dict=response_dict, status_code=200)
    return self.ex_get_network(name=name)