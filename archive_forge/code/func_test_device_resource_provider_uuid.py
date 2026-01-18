from unittest import mock
import uuid
from neutron_lib.placement import utils as place_utils
from neutron_lib.tests import _base as base
def test_device_resource_provider_uuid(self):
    try:
        place_utils.device_resource_provider_uuid(namespace=self._uuid_ns, host='some host', device='some device')
    except Exception:
        self.fail('could not generate device resource provider uuid')