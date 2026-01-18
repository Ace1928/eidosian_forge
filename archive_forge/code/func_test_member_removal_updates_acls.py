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
def test_member_removal_updates_acls(self):
    self.image_stub.locations = [{'url': 'glug', 'metadata': {}, 'status': 'active'}]
    self.image_stub.visibility = 'private'
    membership = glance.domain.ImageMembership(UUID1, TENANT1, None, None, status='accepted')
    self.image_member_repo.remove(membership)
    self.assertIn('glug', self.store_api.acls)
    acls = self.store_api.acls['glug']
    self.assertFalse(acls['public'])
    self.assertEqual([], acls['write'])
    self.assertEqual([TENANT2], acls['read'])