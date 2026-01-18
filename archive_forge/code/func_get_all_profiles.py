import logging
import os
import urllib.parse as urlparse
import urllib.request as urllib
from oslo_vmware import service
from oslo_vmware import vim_util
def get_all_profiles(session):
    """Get all the profiles defined in VC server.

    :returns: PbmProfile data objects
    :raises: VimException, VimFaultException, VimAttributeException,
             VimSessionOverLoadException, VimConnectionException
    """
    LOG.debug('Fetching all the profiles defined in VC server.')
    pbm = session.pbm
    profile_manager = pbm.service_content.profileManager
    res_type = pbm.client.factory.create('ns0:PbmProfileResourceType')
    res_type.resourceType = 'STORAGE'
    profiles = []
    profile_ids = session.invoke_api(pbm, 'PbmQueryProfile', profile_manager, resourceType=res_type)
    LOG.debug('Fetched profile IDs: %s.', profile_ids)
    if profile_ids:
        profiles = session.invoke_api(pbm, 'PbmRetrieveContent', profile_manager, profileIds=profile_ids)
    return profiles