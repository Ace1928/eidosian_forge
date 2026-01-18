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
@staticmethod
def log_http_response(resp, body=None):
    status = (resp.version / 10.0, resp.status, resp.reason)
    dump = ['\nHTTP/%.1f %s %s' % status]
    dump.extend(['%s: %s' % (k, v) for k, v in resp.getheaders()])
    dump.append('')
    if body:
        dump.extend([body, ''])
    LOG.debug('\n'.join(dump))