import copy
from unittest import mock
from oslo_i18n import fixture as i18n_fixture
import suds
from oslo_vmware import exceptions
from oslo_vmware.tests import base
from oslo_vmware import vim
def test_configure_ipv6_and_non_default_host_port(self):
    vim_obj = vim.Vim('https', '::1', 12345)
    self.assertEqual('https://[::1]:12345/sdk/vimService.wsdl', vim_obj.wsdl_url)
    self.assertEqual('https://[::1]:12345/sdk', vim_obj.soap_url)