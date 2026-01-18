from unittest import mock
import uuid
from neutron_lib.placement import utils as place_utils
from neutron_lib.tests import _base as base
def test_agent_resource_provider_uuid(self):
    try:
        place_utils.agent_resource_provider_uuid(namespace=self._uuid_ns, host='some host')
    except Exception:
        self.fail('could not generate agent resource provider uuid')