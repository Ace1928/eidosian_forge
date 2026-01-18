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
def test_delete_image_properties_when_quota_exceeded(self):
    self._quota_exceed_setup()
    self.config(image_property_quota=1)
    del self.image.extra_properties['foo']
    self.image_repo_proxy.save(self.image)
    self.image_repo_mock.save.assert_called_once_with(self.base_image, from_state=None)
    self.assertNotIn('foo', self.base_image.extra_properties)
    self.assertEqual('ham', self.base_image.extra_properties['spam'])