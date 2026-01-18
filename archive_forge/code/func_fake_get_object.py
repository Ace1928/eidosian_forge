import copy
from unittest import mock
import fixtures
import hashlib
import http.client
import importlib
import io
import tempfile
import uuid
from oslo_config import cfg
from oslo_utils import encodeutils
from oslo_utils.secretutils import md5
from oslo_utils import units
import requests_mock
import swiftclient
from glance_store._drivers.swift import connection_manager as manager
from glance_store._drivers.swift import store as swift
from glance_store._drivers.swift import utils as sutils
from glance_store import capabilities
from glance_store import exceptions
from glance_store import location
import glance_store.multi_backend as store
from glance_store.tests import base
from glance_store.tests.unit import test_store_capabilities
def fake_get_object(conn, container, name, **kwargs):
    fixture_key = '%s/%s' % (container, name)
    if fixture_key not in fixture_headers:
        msg = 'Object GET failed'
        status = http.client.NOT_FOUND
        raise swiftclient.ClientException(msg, http_status=status)
    byte_range = None
    headers = kwargs.get('headers', dict())
    if headers is not None:
        headers = dict(((k.lower(), v) for k, v in headers.items()))
        if 'range' in headers:
            byte_range = headers.get('range')
    fixture = fixture_headers[fixture_key]
    if 'manifest' in fixture:
        chunk_keys = sorted([k for k in fixture_headers.keys() if k.startswith(fixture_key) and k != fixture_key])
        result = io.BytesIO()
        for key in chunk_keys:
            result.write(fixture_objects[key].getvalue())
    else:
        result = fixture_objects[fixture_key]
    if byte_range is not None:
        start = int(byte_range.split('=')[1].strip('-'))
        result = io.BytesIO(result.getvalue()[start:])
        fixture_headers[fixture_key]['content-length'] = len(result.getvalue())
    return (fixture_headers[fixture_key], result)