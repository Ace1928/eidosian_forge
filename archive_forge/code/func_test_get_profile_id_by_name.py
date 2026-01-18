import os
from unittest import mock
import urllib.parse as urlparse
import urllib.request as urllib
from oslo_vmware import pbm
from oslo_vmware.tests import base
from oslo_vmware import vim_util
@mock.patch.object(pbm, 'get_all_profiles')
def test_get_profile_id_by_name(self, get_all_profiles):
    profiles = [self._create_profile(str(i), 'profile-%d' % i) for i in range(0, 10)]
    get_all_profiles.return_value = profiles
    session = mock.Mock()
    exp_profile_id = '5'
    profile_id = pbm.get_profile_id_by_name(session, 'profile-%s' % exp_profile_id)
    self.assertEqual(exp_profile_id, profile_id)
    get_all_profiles.assert_called_once_with(session)