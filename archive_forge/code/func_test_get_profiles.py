import os
from unittest import mock
import urllib.parse as urlparse
import urllib.request as urllib
from oslo_vmware import pbm
from oslo_vmware.tests import base
from oslo_vmware import vim_util
def test_get_profiles(self):
    pbm_service = mock.Mock()
    session = mock.Mock(pbm=pbm_service)
    object_ref = mock.Mock()
    pbm_service.client.factory.create.return_value = object_ref
    profile_id = mock.sentinel.profile_id
    session.invoke_api.return_value = profile_id
    value = 'vm-1'
    vm = vim_util.get_moref(value, 'VirtualMachine')
    ret = pbm.get_profiles(session, vm)
    self.assertEqual(profile_id, ret)
    session.invoke_api.assert_called_once_with(pbm_service, 'PbmQueryAssociatedProfile', pbm_service.service_content.profileManager, entity=object_ref)
    self.assertEqual(value, object_ref.key)
    self.assertEqual('virtualMachine', object_ref.objectType)