import copy
from unittest import mock
from oslo_i18n import fixture as i18n_fixture
import suds
from oslo_vmware import exceptions
from oslo_vmware.tests import base
from oslo_vmware import vim
def test_configure_with_wsdl_url_override(self):
    vim_obj = vim.Vim('https', 'www.example.com', wsdl_url='https://test.com/sdk/vimService.wsdl')
    self.assertEqual('https://test.com/sdk/vimService.wsdl', vim_obj.wsdl_url)
    self.assertEqual('https://www.example.com/sdk', vim_obj.soap_url)