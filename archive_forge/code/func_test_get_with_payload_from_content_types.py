from unittest import mock
from openstack.key_manager.v1 import secret
from openstack.tests.unit import base
def test_get_with_payload_from_content_types(self):
    content_type = 'some/type'
    metadata = {'status': 'fine', 'content_types': {'default': content_type}}
    sot = secret.Secret(id='id')
    self._test_payload(sot, metadata, content_type)