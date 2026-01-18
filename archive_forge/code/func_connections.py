from __future__ import absolute_import
import logging
from google.auth import exceptions
from google.auth import transport
import httplib2
from six.moves import http_client
@connections.setter
def connections(self, value):
    """Proxy to httplib2.Http.connections."""
    self.http.connections = value