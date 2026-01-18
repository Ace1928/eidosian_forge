from urllib import parse
import ddt
from cinderclient import api_versions
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
def test_delete_messages(self):
    cs = fakes.FakeClient(api_versions.APIVersion('3.3'))
    fake_id = '1234'
    cs.messages.delete(fake_id)
    cs.assert_called('DELETE', '/messages/%s' % fake_id)