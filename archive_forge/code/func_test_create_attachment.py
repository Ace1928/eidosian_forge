from cinderclient import api_versions
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
def test_create_attachment(self):
    cs = fakes.FakeClient(api_versions.APIVersion('3.27'))
    att = cs.attachments.create('e84fda45-4de4-4ce4-8f39-fc9d3b0aa05e', {}, '557ad76c-ce54-40a3-9e91-c40d21665cc3', 'null')
    cs.assert_called('POST', '/attachments')
    self.assertEqual(fakes.fake_attachment['attachment'], att)