import sys
from unittest import mock
import urllib.error
from glance_store import exceptions as store_exceptions
from oslo_config import cfg
from oslo_utils import units
import taskflow
import glance.async_.flows.api_image_import as import_flow
from glance.common import exception
from glance.common.scripts.image_import import main as image_import
from glance import context
from glance.domain import ExtraProperties
from glance import gateway
import glance.tests.utils as test_utils
from cursive import exception as cursive_exception
def test_image_property_proxies(self):
    self.assertEqual(IMAGE_ID1, self.actions.image_id)
    self.assertEqual('active', self.actions.image_status)
    self.assertEqual('raw', self.actions.image_disk_format)
    self.assertEqual('bare', self.actions.image_container_format)
    self.assertEqual({'speed': '88mph'}, self.actions.image_extra_properties)