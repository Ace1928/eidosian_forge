import ssl
from unittest import mock
import requests
from oslo_vmware import exceptions
from oslo_vmware import rw_handles
from oslo_vmware.tests import base
from oslo_vmware import vim_util
def session_invoke_api_side_effect(module, method, *args, **kwargs):
    if module == vim_util and method == 'get_object_property':
        return 'ready'
    self.assertEqual(session.vim, module)
    self.assertEqual('HttpNfcLeaseComplete', method)