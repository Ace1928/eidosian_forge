import http.client as httplib
import io
from unittest import mock
import ddt
import requests
import suds
from oslo_vmware import exceptions
from oslo_vmware import service
from oslo_vmware.tests import base
from oslo_vmware import vim_util
def test_filter_with_login_failure(self):
    message = mock.Mock(spec=suds.sax.element.Element)

    def child_at_path_mock(path):
        if path == '/Envelope/Body/Login':
            return self.login
    message.childAtPath.side_effect = child_at_path_mock
    record = mock.Mock(msg=message)
    self.assertTrue(self.log_filter.filter(record))
    self.assertEqual('***', self.username.getText())
    self.assertEqual('***', self.password.getText())
    self.assertEqual('bcdef', self.session_id.getText())