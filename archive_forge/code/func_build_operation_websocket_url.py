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
def build_operation_websocket_url(self, uuid, w_secret):
    uri = 'wss://%s:%s/%s/operations/%s/websocket?secret=%s' % (self.connection.host, self.connection.port, self.version, uuid, w_secret)
    return uri