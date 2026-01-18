import datetime
from unittest import mock
import uuid
from oslo_serialization import jsonutils as json
from oslo_utils import timeutils
from heat.common import identifier
from heat.common import template_format
from heat.engine.clients.os import heat_plugin
from heat.engine.clients.os import swift as swift_plugin
from heat.engine import environment
from heat.engine import resource
from heat.engine.resources.openstack.heat import wait_condition_handle as h_wch
from heat.engine import stack as parser
from heat.engine import template as tmpl
from heat.objects import resource as resource_objects
from heat.tests import common
from heat.tests import utils
def test_getatt_signal_swift(self):

    class mock_swift(object):

        @staticmethod
        def put_container(container, **kwargs):
            pass

        @staticmethod
        def put_object(container, object, contents, **kwargs):
            pass
    mock_tempurl = self.patchobject(swift_plugin.SwiftClientPlugin, 'get_temp_url', return_value='foo')
    self.patchobject(swift_plugin.SwiftClientPlugin, 'client', return_value=mock_swift)
    handle = self._create_heat_handle(template=test_template_heat_waithandle_swift)
    self.assertIsNone(handle.FnGetAtt('token'))
    self.assertIsNone(handle.FnGetAtt('endpoint'))
    self.assertIsNone(handle.FnGetAtt('curl_cli'))
    signal = json.loads(handle.FnGetAtt('signal'))
    self.assertIn('alarm_url', signal)
    mock_tempurl.assert_called_once()