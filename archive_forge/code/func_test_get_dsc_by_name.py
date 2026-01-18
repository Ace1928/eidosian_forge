from unittest import mock
from oslo_utils import units
import urllib.parse as urlparse
from oslo_vmware import constants
from oslo_vmware.objects import datastore
from oslo_vmware.tests import base
from oslo_vmware import vim_util
@mock.patch('oslo_vmware.vim_util.continue_retrieval')
@mock.patch('oslo_vmware.vim_util.cancel_retrieval')
def test_get_dsc_by_name(self, cancel_retrieval, continue_retrieval):
    pod_prop = mock.Mock()
    pod_prop.val = 'ds-cluster'
    pod_ref = vim_util.get_moref('group-p456', 'StoragePod')
    pod = mock.Mock()
    pod.propSet = [pod_prop]
    pod.obj = pod_ref
    retrieve_result = mock.Mock()
    retrieve_result.objects = [pod]
    session = mock.Mock()
    session.invoke_api = mock.Mock()
    session.invoke_api.return_value = retrieve_result
    name = 'ds-cluster'
    dsc_ref, dsc_name = datastore.get_dsc_ref_and_name(session, name)
    self.assertEqual((vim_util.get_moref_value(pod_ref), vim_util.get_moref_type(pod_ref)), (vim_util.get_moref_value(dsc_ref), vim_util.get_moref_type(dsc_ref)))