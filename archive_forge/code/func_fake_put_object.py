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
def fake_put_object(url, token, container, name, contents, **kwargs):
    global SWIFT_PUT_OBJECT_CALLS
    SWIFT_PUT_OBJECT_CALLS += 1
    CHUNKSIZE = 64 * units.Ki
    fixture_key = '%s/%s' % (container, name)
    if fixture_key not in fixture_headers:
        if kwargs.get('headers'):
            manifest = kwargs.get('headers').get('X-Object-Manifest')
            etag = kwargs.get('headers').get('ETag', md5(b'', usedforsecurity=False).hexdigest())
            fixture_headers[fixture_key] = {'manifest': True, 'etag': etag, 'x-object-manifest': manifest}
            fixture_objects[fixture_key] = None
            return etag
        if hasattr(contents, 'read'):
            fixture_object = io.BytesIO()
            read_len = 0
            chunk = contents.read(CHUNKSIZE)
            checksum = md5(usedforsecurity=False)
            while chunk:
                fixture_object.write(chunk)
                read_len += len(chunk)
                checksum.update(chunk)
                chunk = contents.read(CHUNKSIZE)
            etag = checksum.hexdigest()
        else:
            fixture_object = io.BytesIO(contents)
            read_len = len(contents)
            etag = md5(fixture_object.getvalue(), usedforsecurity=False).hexdigest()
        if read_len > MAX_SWIFT_OBJECT_SIZE:
            msg = 'Image size:%d exceeds Swift max:%d' % (read_len, MAX_SWIFT_OBJECT_SIZE)
            raise swiftclient.ClientException(msg, http_status=http.client.REQUEST_ENTITY_TOO_LARGE)
        fixture_objects[fixture_key] = fixture_object
        fixture_headers[fixture_key] = {'content-length': read_len, 'etag': etag}
        return etag
    else:
        msg = 'Object PUT failed - Object with key %s already exists' % fixture_key
        raise swiftclient.ClientException(msg, http_status=http.client.CONFLICT)