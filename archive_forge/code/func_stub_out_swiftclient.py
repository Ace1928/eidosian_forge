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
def stub_out_swiftclient(self, swift_store_auth_version):
    fixture_containers = ['glance']
    fixture_container_headers = {}
    fixture_headers = {'glance/%s' % FAKE_UUID: {'content-length': FIVE_KB, 'etag': 'c2e5db72bd7fd153f53ede5da5a06de3'}, 'glance/%s' % FAKE_UUID2: {'x-static-large-object': 'true'}}
    fixture_objects = {'glance/%s' % FAKE_UUID: io.BytesIO(b'*' * FIVE_KB), 'glance/%s' % FAKE_UUID2: io.BytesIO(b'*' * FIVE_KB)}

    def fake_head_container(url, token, container, **kwargs):
        if container not in fixture_containers:
            msg = 'No container %s found' % container
            status = http.client.NOT_FOUND
            raise swiftclient.ClientException(msg, http_status=status)
        return fixture_container_headers

    def fake_put_container(url, token, container, **kwargs):
        fixture_containers.append(container)

    def fake_post_container(url, token, container, headers, **kwargs):
        for key, value in headers.items():
            fixture_container_headers[key] = value

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

    def fake_head_object(url, token, container, name, **kwargs):
        try:
            fixture_key = '%s/%s' % (container, name)
            return fixture_headers[fixture_key]
        except KeyError:
            msg = 'Object HEAD failed - Object does not exist'
            status = http.client.NOT_FOUND
            raise swiftclient.ClientException(msg, http_status=status)

    def fake_delete_object(url, token, container, name, **kwargs):
        fixture_key = '%s/%s' % (container, name)
        if fixture_key not in fixture_headers:
            msg = 'Object DELETE failed - Object does not exist'
            status = http.client.NOT_FOUND
            raise swiftclient.ClientException(msg, http_status=status)
        else:
            del fixture_headers[fixture_key]
            del fixture_objects[fixture_key]

    def fake_http_connection(*args, **kwargs):
        return None

    def fake_get_auth(url, user, key, auth_version, **kwargs):
        if url is None:
            return (None, None)
        if 'http' in url and '://' not in url:
            raise ValueError('Invalid url %s' % url)
        if swift_store_auth_version != auth_version:
            msg = 'AUTHENTICATION failed (version mismatch)'
            raise swiftclient.ClientException(msg)
        return (None, None)
    self.useFixture(fixtures.MockPatch('swiftclient.client.head_container', fake_head_container))
    self.useFixture(fixtures.MockPatch('swiftclient.client.put_container', fake_put_container))
    self.useFixture(fixtures.MockPatch('swiftclient.client.post_container', fake_post_container))
    self.useFixture(fixtures.MockPatch('swiftclient.client.put_object', fake_put_object))
    self.useFixture(fixtures.MockPatch('swiftclient.client.delete_object', fake_delete_object))
    self.useFixture(fixtures.MockPatch('swiftclient.client.head_object', fake_head_object))
    self.useFixture(fixtures.MockPatch('swiftclient.client.Connection.get_object', fake_get_object))
    self.useFixture(fixtures.MockPatch('swiftclient.client.get_auth', fake_get_auth))
    self.useFixture(fixtures.MockPatch('swiftclient.client.http_connection', fake_http_connection))