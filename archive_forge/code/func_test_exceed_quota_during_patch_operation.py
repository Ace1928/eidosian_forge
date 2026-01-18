import copy
import fixtures
from unittest import mock
from unittest.mock import patch
import uuid
from oslo_limit import exception as ol_exc
from oslo_utils import encodeutils
from oslo_utils import units
from glance.common import exception
from glance.common import store_utils
import glance.quota
from glance.quota import keystone as ks_quota
from glance.tests.unit import fixtures as glance_fixtures
from glance.tests.unit import utils as unit_test_utils
from glance.tests import utils as test_utils
def test_exceed_quota_during_patch_operation(self):
    self._quota_exceed_setup()
    self.image.extra_properties['frob'] = 'baz'
    self.image.extra_properties['lorem'] = 'ipsum'
    self.assertEqual('bar', self.base_image.extra_properties['foo'])
    self.assertEqual('ham', self.base_image.extra_properties['spam'])
    self.assertEqual('baz', self.base_image.extra_properties['frob'])
    self.assertEqual('ipsum', self.base_image.extra_properties['lorem'])
    del self.image.extra_properties['frob']
    del self.image.extra_properties['lorem']
    self.image_repo_proxy.save(self.image)
    call_args = mock.call(self.base_image, from_state=None)
    self.assertEqual(call_args, self.image_repo_mock.save.call_args)
    self.assertEqual('bar', self.base_image.extra_properties['foo'])
    self.assertEqual('ham', self.base_image.extra_properties['spam'])
    self.assertNotIn('frob', self.base_image.extra_properties)
    self.assertNotIn('lorem', self.base_image.extra_properties)