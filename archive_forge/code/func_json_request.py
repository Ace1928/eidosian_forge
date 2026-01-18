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
def json_request(self, method, url, **kwargs):
    kwargs.setdefault('headers', {})
    kwargs['headers'].setdefault('Content-Type', 'application/json')
    kwargs['headers'].setdefault('Accept', 'application/json')
    if 'body' in kwargs:
        kwargs['data'] = jsonutils.dumps(kwargs.pop('body'))
    resp = self._http_request(url, method, **kwargs)
    body = resp.content
    content_type = resp.headers.get('content-type', None)
    status = resp.status_code
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