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
def test_filter_with_unknown_failure(self):
    message = mock.Mock(spec=suds.sax.element.Element)

    def child_at_path_mock(path):
        return None
    message.childAtPath.side_effect = child_at_path_mock
    record = mock.Mock(msg=message)
    self.assertTrue(self.log_filter.filter(record))
    self.assertEqual('admin', self.username.getText())
    self.assertEqual('password', self.password.getText())
    self.assertEqual('abcdef', self.session_id.getText())