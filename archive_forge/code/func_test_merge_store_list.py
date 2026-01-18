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
def test_merge_store_list(self):
    self.actions.merge_store_list('stores', ['foo', 'bar'])
    self.assertEqual({'speed': '88mph', 'stores': 'bar,foo'}, self.image.extra_properties)
    self.actions.merge_store_list('stores', ['baz'])
    self.assertEqual('bar,baz,foo', self.image.extra_properties['stores'])
    self.actions.merge_store_list('stores', ['foo'], subtract=True)
    self.assertEqual('bar,baz', self.image.extra_properties['stores'])
    self.actions.merge_store_list('stores', ['bar'])
    self.assertEqual('bar,baz', self.image.extra_properties['stores'])
    self.actions.merge_store_list('stores', ['baz', 'bar'], subtract=True)
    self.assertEqual('', self.image.extra_properties['stores'])
    self.actions.merge_store_list('stores', ['', None])
    self.assertEqual('', self.image.extra_properties['stores'])