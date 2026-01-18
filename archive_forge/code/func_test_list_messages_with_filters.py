from urllib import parse
import ddt
from cinderclient import api_versions
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
@ddt.data('id', 'resource_type', 'event_id', 'resource_uuid', 'message_level', 'guaranteed_until', 'request_id')
def test_list_messages_with_filters(self, filter_string):
    cs = fakes.FakeClient(api_versions.APIVersion('3.5'))
    cs.messages.list(search_opts={filter_string: 'value'})
    cs.assert_called('GET', '/messages?%s=value' % parse.quote(filter_string))