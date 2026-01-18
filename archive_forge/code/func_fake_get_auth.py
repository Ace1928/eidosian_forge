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
def fake_get_auth(url, user, key, auth_version, **kwargs):
    if url is None:
        return (None, None)
    if 'http' in url and '://' not in url:
        raise ValueError('Invalid url %s' % url)
    if swift_store_auth_version != auth_version:
        msg = 'AUTHENTICATION failed (version mismatch)'
        raise swiftclient.ClientException(msg)
    return (None, None)