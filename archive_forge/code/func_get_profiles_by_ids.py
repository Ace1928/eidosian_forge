import logging
import os
import urllib.parse as urlparse
import urllib.request as urllib
from oslo_vmware import service
from oslo_vmware import vim_util
def get_profiles_by_ids(session, profile_ids):
    """Get storage profiles by IDs.

    :param session: VMwareAPISession instance
    :param profile_ids: profile IDs
    :return: profile objects
    """
    profiles = []
    if profile_ids:
        pbm = session.pbm
        profile_manager = pbm.service_content.profileManager
        profiles = session.invoke_api(pbm, 'PbmRetrieveContent', profile_manager, profileIds=profile_ids)
    return profiles