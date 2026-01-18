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
def raw_request(self, method, url, **kwargs):
    kwargs.setdefault('headers', {})
    kwargs['headers'].setdefault('Content-Type', 'application/octet-stream')
    resp = self._http_request(url, method, **kwargs)
    body = resp.content
    status = resp.status_code
    content_type = resp.headers.get('content-type', None)
    if status == 204 or status == 205 or content_type is None:
        return (resp, list())
    if 'application/json' in content_type:
        try:
            body = resp.json()
        except ValueError:
            LOG.error('Could not decode response body as JSON')
    else:
        body = None
    return (resp, body)