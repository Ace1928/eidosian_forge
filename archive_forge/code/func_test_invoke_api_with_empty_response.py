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
def test_invoke_api_with_empty_response(self):
    api_session = self._create_api_session(True)
    vim_obj = api_session.vim
    vim_obj.SessionIsActive.return_value = True

    def api(*args, **kwargs):
        raise exceptions.VimFaultException([exceptions.NOT_AUTHENTICATED], None)
    module = mock.Mock()
    module.api = api
    ret = api_session.invoke_api(module, 'api')
    self.assertEqual([], ret)
    vim_obj.SessionIsActive.assert_called_once_with(vim_obj.service_content.sessionManager, sessionID=api_session._session_id, userName=api_session._session_username)