from datetime import datetime
from unittest import mock
from eventlet import greenthread
from oslo_context import context
import suds
from oslo_vmware import api
from oslo_vmware import exceptions
from oslo_vmware import pbm
from oslo_vmware.tests import base
from oslo_vmware import vim_util
def test_logout_calls_vim_logout(self):
    session = mock.Mock()
    session.key = '12345'
    api_session = self._create_api_session(False)
    vim_obj = api_session.vim
    vim_obj.Login.return_value = session
    vim_obj.Logout.return_value = None
    api_session._create_session()
    session_manager = vim_obj.service_content.sessionManager
    vim_obj.Login.assert_called_once_with(session_manager, userName=VMwareAPISessionTest.USERNAME, password=VMwareAPISessionTest.PASSWORD, locale='en')
    api_session.logout()
    vim_obj.Logout.assert_called_once_with(session_manager)
    self.assertIsNone(api_session._session_id)