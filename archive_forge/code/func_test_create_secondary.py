import time
import uuid
from designateclient.tests import v2
def test_create_secondary(self):
    ref = self.new_ref(type='SECONDARY', masters=['10.0.0.1'])
    self.stub_url('POST', parts=[self.RESOURCE], json=ref)
    values = ref.copy()
    del values['id']
    self.client.zones.create(values['name'], type_=values['type'], masters=values['masters'])
    self.assertRequestBodyIs(json=values)