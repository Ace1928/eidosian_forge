from cryptography import exceptions as crypto_exception
from cursive import exception as cursive_exception
from cursive import signature_utils
import glance_store
from unittest import mock
from glance.common import exception
import glance.location
from glance.tests.unit import base as unit_test_base
from glance.tests.unit import utils as unit_test_utils
from glance.tests import utils
def test_add_updates_acls(self):
    self.image_stub.locations = [{'url': 'foo', 'metadata': {}, 'status': 'active'}, {'url': 'bar', 'metadata': {}, 'status': 'active'}]
    self.image_stub.visibility = 'public'
    self.image_repo.add(self.image)
    self.assertTrue(self.store_api.acls['foo']['public'])
    self.assertEqual([], self.store_api.acls['foo']['read'])
    self.assertEqual([], self.store_api.acls['foo']['write'])
    self.assertTrue(self.store_api.acls['bar']['public'])
    self.assertEqual([], self.store_api.acls['bar']['read'])
    self.assertEqual([], self.store_api.acls['bar']['write'])