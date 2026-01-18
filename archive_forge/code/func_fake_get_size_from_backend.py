import glance_store
from unittest import mock
from glance.common import exception
from glance.common import store_utils
import glance.location
from glance.tests.unit import base
def fake_get_size_from_backend(uri, context=None):
    return 1