import os
from unittest import mock
import glance_store as store
from glance_store._drivers import cinder
from glance_store._drivers import rbd as rbd_store
from glance_store._drivers import swift
from glance_store import location
from oslo_concurrency import lockutils
from oslo_config import cfg
from oslo_db import options
from oslo_serialization import jsonutils
from glance.tests import stubs
from glance.tests import utils as test_utils
def fake_get_conection_type(client):
    DEFAULT_API_PORT = 9292
    if client.port == DEFAULT_API_PORT:
        return stubs.FakeGlanceConnection