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
def test_enforce_underquota_with_others_over_quota(self):
    self.config(endpoint_id='ENDPOINT_ID', group='oslo_limit')
    self.config(use_keystone_limits=True)
    context = FakeContext()
    self._create_fake_image(context, 300)
    self._create_fake_image(context, 300)
    other_context = FakeContext()
    other_context.owner = 'someone_else'
    self._create_fake_image(other_context, 100)
    ks_quota.enforce_image_size_total(other_context, other_context.owner)