from urllib import parse
import ddt
from cinderclient import api_versions
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
def test_list_messages(self):
    cs = fakes.FakeClient(api_versions.APIVersion('3.3'))
    cs.messages.list()
    cs.assert_called('GET', '/messages')