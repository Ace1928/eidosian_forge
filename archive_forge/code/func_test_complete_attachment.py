from cinderclient import api_versions
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
def test_complete_attachment(self):
    cs = fakes.FakeClient(api_versions.APIVersion('3.44'))
    att = cs.attachments.complete('a232e9ae')
    self.assertTrue(att.ok)