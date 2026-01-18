import logging
import os
import urllib.parse as urlparse
import urllib.request as urllib
from oslo_vmware import service
from oslo_vmware import vim_util
def filter_hubs_by_profile(session, hubs, profile_id):
    """Filter and return hubs that match the given profile.

    :param hubs: PbmPlacementHub morefs
    :param profile_id: profile ID
    :returns: subset of hubs that match the given profile
    :raises: VimException, VimFaultException, VimAttributeException,
             VimSessionOverLoadException, VimConnectionException
    """
    LOG.debug('Filtering hubs: %(hubs)s that match profile: %(profile)s.', {'hubs': hubs, 'profile': profile_id})
    pbm = session.pbm
    placement_solver = pbm.service_content.placementSolver
    filtered_hubs = session.invoke_api(pbm, 'PbmQueryMatchingHub', placement_solver, hubsToSearch=hubs, profile=profile_id)
    LOG.debug('Filtered hubs: %s', filtered_hubs)
    return filtered_hubs