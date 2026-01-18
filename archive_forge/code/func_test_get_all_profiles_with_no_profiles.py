import os
from unittest import mock
import urllib.parse as urlparse
import urllib.request as urllib
from oslo_vmware import pbm
from oslo_vmware.tests import base
from oslo_vmware import vim_util
def test_get_all_profiles_with_no_profiles(self):
    session = mock.Mock()
    session.pbm = mock.Mock()
    session.invoke_api.return_value = []
    profiles = pbm.get_all_profiles(session)
    session.invoke_api.assert_called_once_with(session.pbm, 'PbmQueryProfile', session.pbm.service_content.profileManager, resourceType=session.pbm.client.factory.create())
    self.assertEqual([], profiles)