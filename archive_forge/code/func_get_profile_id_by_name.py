import logging
import os
import urllib.parse as urlparse
import urllib.request as urllib
from oslo_vmware import service
from oslo_vmware import vim_util
def get_profile_id_by_name(session, profile_name):
    """Get the profile UUID corresponding to the given profile name.

    :param profile_name: profile name whose UUID needs to be retrieved
    :returns: profile UUID string or None if profile not found
    :raises: VimException, VimFaultException, VimAttributeException,
             VimSessionOverLoadException, VimConnectionException
    """
    LOG.debug('Retrieving profile ID for profile: %s.', profile_name)
    for profile in get_all_profiles(session):
        if profile.name == profile_name:
            profile_id = profile.profileId
            LOG.debug('Retrieved profile ID: %(id)s for profile: %(name)s.', {'id': profile_id, 'name': profile_name})
            return profile_id
    return None