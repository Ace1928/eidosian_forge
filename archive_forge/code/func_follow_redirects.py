from __future__ import absolute_import
import logging
from google.auth import exceptions
from google.auth import transport
import httplib2
from six.moves import http_client
@follow_redirects.setter
def follow_redirects(self, value):
    """Proxy to httplib2.Http.follow_redirects."""
    self.http.follow_redirects = value