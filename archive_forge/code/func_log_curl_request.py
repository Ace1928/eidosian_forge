import copy
from http import client as http_client
import io
import logging
import os
import socket
import ssl
from urllib import parse as urlparse
from keystoneauth1 import adapter
from oslo_serialization import jsonutils
from oslo_utils import importutils
from magnumclient import exceptions
def log_curl_request(self, method, url, kwargs):
    curl = ['curl -i -X %s' % method]
    for key, value in kwargs['headers'].items():
        header = "-H '%s: %s'" % (key, value)
        curl.append(header)
    conn_params_fmt = [('key_file', '--key %s'), ('cert_file', '--cert %s'), ('ca_file', '--cacert %s')]
    for key, fmt in conn_params_fmt:
        value = self.connection_params[2].get(key)
        if value:
            curl.append(fmt % value)
    if self.connection_params[2].get('insecure'):
        curl.append('-k')
    if 'body' in kwargs:
        curl.append("-d '%s'" % kwargs['body'])
    curl.append('%s/%s' % (self.endpoint, url.lstrip(API_VERSION)))
    LOG.debug(' '.join(curl))