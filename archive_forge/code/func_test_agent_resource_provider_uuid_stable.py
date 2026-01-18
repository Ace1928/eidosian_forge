from unittest import mock
import uuid
from neutron_lib.placement import utils as place_utils
from neutron_lib.tests import _base as base
def test_agent_resource_provider_uuid_stable(self):
    uuid_a = place_utils.agent_resource_provider_uuid(namespace=self._uuid_ns, host='somehost')
    uuid_b = place_utils.agent_resource_provider_uuid(namespace=self._uuid_ns, host='somehost')
    self.assertEqual(uuid_a, uuid_b)