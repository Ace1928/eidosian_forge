from cinderclient import api_versions
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
def test_create_attachment_without_instance_uuid(self):
    cs = fakes.FakeClient(api_versions.APIVersion('3.27'))
    att = cs.attachments.create('e84fda45-4de4-4ce4-8f39-fc9d3b0aa05e', {}, None, 'null')
    cs.assert_called('POST', '/attachments')
    self.assertEqual(fakes.fake_attachment_without_instance_id['attachment'], att)