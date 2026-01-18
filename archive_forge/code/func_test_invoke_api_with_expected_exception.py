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
def test_invoke_api_with_expected_exception(self):
    api_session = self._create_api_session(True)
    api_session._create_session = mock.Mock()
    vim_obj = api_session.vim
    vim_obj.SessionIsActive.return_value = False
    ret = mock.Mock()
    responses = [exceptions.VimConnectionException(None), ret]

    def api(*args, **kwargs):
        response = responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response
    module = mock.Mock()
    module.api = api
    with mock.patch.object(greenthread, 'sleep'):
        self.assertEqual(ret, api_session.invoke_api(module, 'api'))
    api_session._create_session.assert_called_once_with()