from unittest import mock
from openstack.key_manager.v1 import secret
from openstack.tests.unit import base
def test_get_with_payload_from_argument(self):
    metadata = {'status': 'great'}
    content_type = 'some/type'
    sot = secret.Secret(id='id', payload_content_type=content_type)
    self._test_payload(sot, metadata, content_type)