import datetime
from unittest import mock
import uuid
import fixtures
import freezegun
import http.client
from oslo_config import fixture as config_fixture
from oslo_log import log
import oslo_messaging
from pycadf import cadftaxonomy
from pycadf import cadftype
from pycadf import eventfactory
from pycadf import resource as cadfresource
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone import notifications
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_a_list_of_callbacks(self):

    def callback(*args, **kwargs):
        pass

    class C(object):

        def callback(self, *args, **kwargs):
            pass
    with mock.patch('keystone.notifications.LOG', self.mock_log):
        notifications.register_event_callback(CREATED_OPERATION, 'thing', [callback, C().callback])
    callback_1 = 'keystone.tests.unit.common.test_notifications.callback'
    callback_2 = 'keystone.tests.unit.common.test_notifications.C.callback'
    expected_log_data = [{'callback': callback_1, 'event': 'identity.thing.created'}, {'callback': callback_2, 'event': 'identity.thing.created'}]
    self.verify_log_message(expected_log_data)