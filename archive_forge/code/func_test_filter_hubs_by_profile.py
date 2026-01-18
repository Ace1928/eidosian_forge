import os
from unittest import mock
import urllib.parse as urlparse
import urllib.request as urllib
from oslo_vmware import pbm
from oslo_vmware.tests import base
from oslo_vmware import vim_util
def test_filter_hubs_by_profile(self):
    pbm_client = mock.Mock()
    session = mock.Mock()
    session.pbm = pbm_client
    hubs = mock.Mock()
    profile_id = 'profile-0'
    pbm.filter_hubs_by_profile(session, hubs, profile_id)
    session.invoke_api.assert_called_once_with(pbm_client, 'PbmQueryMatchingHub', pbm_client.service_content.placementSolver, hubsToSearch=hubs, profile=profile_id)