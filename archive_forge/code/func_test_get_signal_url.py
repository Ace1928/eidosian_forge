import datetime
from unittest import mock
from testtools import matchers
from heat.engine.clients.os import swift
from heat.tests import common
from heat.tests import utils
def test_get_signal_url(self):
    self.swift_client.url = 'http://fake-host.com:8080/v1/AUTH_demo'
    self.swift_client.head_account = mock.Mock(return_value={'x-account-meta-temp-url-key': '123456'})
    self.swift_client.post_account = mock.Mock()
    container_name = '1234'
    stack_name = 'test'
    handle_name = 'foo'
    obj_name = '%s-%s' % (stack_name, handle_name)
    url = self.swift_plugin.get_signal_url(container_name, obj_name)
    self.assertTrue(self.swift_client.put_container.called)
    self.assertTrue(self.swift_client.put_object.called)
    regexp = 'http://fake-host.com:8080/v1/AUTH_demo/%s/%s\\?temp_url_sig=[0-9a-f]{40,64}&temp_url_expires=[0-9]{10}' % (container_name, obj_name)
    self.assertThat(url, matchers.MatchesRegex(regexp))