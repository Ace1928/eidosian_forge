import collections
from unittest import mock
from oslo_vmware.tests import base
from oslo_vmware import vim_util
@mock.patch('oslo_vmware.vim_util._get_token')
def test_continue_retrieval(self, get_token):
    token = mock.Mock()
    get_token.return_value = token
    vim = mock.Mock()
    retrieve_result = mock.Mock()
    vim_util.continue_retrieval(vim, retrieve_result)
    get_token.assert_called_once_with(retrieve_result)
    vim.ContinueRetrievePropertiesEx.assert_called_once_with(vim.service_content.propertyCollector, token=token)