from unittest import mock
from uuid import uuid4
import testtools
from openstack.cloud import _utils
from openstack import exceptions
from openstack.tests.unit import base
def test_get_entity_pass_dict(self):
    d = dict(id=uuid4().hex)
    self.cloud.use_direct_get = True
    self.assertEqual(d, _utils._get_entity(self.cloud, '', d, {}))