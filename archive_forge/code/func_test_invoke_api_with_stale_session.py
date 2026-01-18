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
def test_invoke_api_with_stale_session(self):
    api_session = self._create_api_session(True)
    api_session._create_session = mock.Mock()
    vim_obj = api_session.vim
    vim_obj.SessionIsActive.return_value = False
    result = mock.Mock()
    responses = [exceptions.VimFaultException([exceptions.NOT_AUTHENTICATED], None), result]

    def api(*args, **kwargs):
        response = responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response
    module = mock.Mock()
    module.api = api
    with mock.patch.object(greenthread, 'sleep'):
        ret = api_session.invoke_api(module, 'api')
    self.assertEqual(result, ret)
    vim_obj.SessionIsActive.assert_called_once_with(vim_obj.service_content.sessionManager, sessionID=api_session._session_id, userName=api_session._session_username)
    api_session._create_session.assert_called_once_with()