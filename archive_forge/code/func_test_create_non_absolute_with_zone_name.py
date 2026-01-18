from unittest.mock import patch
import uuid
import testtools
from designateclient import exceptions
from designateclient.tests import v2
from designateclient.v2 import zones
@patch.object(zones.ZoneController, 'get')
def test_create_non_absolute_with_zone_name(self, zone_get):
    ref = self.new_ref()
    zone_get.return_value = ZONE
    parts = ['zones', ZONE['id'], self.RESOURCE]
    self.stub_url('POST', parts=parts, json=ref)
    values = ref.copy()
    del values['id']
    self.client.recordsets.create(ZONE['name'], values['name'], values['type'], values['records'])
    values['name'] = '{}.{}'.format(ref['name'], ZONE['name'])
    self.assertRequestBodyIs(json=values)