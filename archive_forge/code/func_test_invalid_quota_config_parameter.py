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
def test_invalid_quota_config_parameter(self):
    self.config(user_storage_quota='foo')
    location = {'url': 'file:///fake.img.tar.gz', 'metadata': {}}
    self.assertRaises(exception.InvalidOptionValue, self.image.locations.append, location)