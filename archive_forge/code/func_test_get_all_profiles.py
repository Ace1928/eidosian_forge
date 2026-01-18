import os
from unittest import mock
import urllib.parse as urlparse
import urllib.request as urllib
from oslo_vmware import pbm
from oslo_vmware.tests import base
from oslo_vmware import vim_util
def test_get_all_profiles(self):
    session = mock.Mock()
    session.pbm = mock.Mock()
    profile_ids = mock.Mock()

    def invoke_api_side_effect(module, method, *args, **kwargs):
        self.assertEqual(session.pbm, module)
        self.assertIn(method, ['PbmQueryProfile', 'PbmRetrieveContent'])
        self.assertEqual(session.pbm.service_content.profileManager, args[0])
        if method == 'PbmQueryProfile':
            self.assertEqual('STORAGE', kwargs['resourceType'].resourceType)
            return profile_ids
        self.assertEqual(profile_ids, kwargs['profileIds'])
    session.invoke_api.side_effect = invoke_api_side_effect
    pbm.get_all_profiles(session)
    self.assertEqual(2, session.invoke_api.call_count)