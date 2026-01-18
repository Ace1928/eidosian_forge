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
def test_get_container_name_with_multiple_containers_turned_off(self):
    self.config(group='swift1', swift_store_multiple_containers_seed=0)
    self.store.configure()
    test_image_id = 'random_id'
    actual = self.store.get_container_name(test_image_id, 'default_container')
    expected = 'default_container'
    self.assertEqual(expected, actual)