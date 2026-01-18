from __future__ import absolute_import, division, print_function
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi
def find_storage_profile_by_name(self, profile_name):
    storage_profile = None
    self.get_spbm_connection()
    pm = self.spbm_content.profileManager
    profile_ids = pm.PbmQueryProfile(resourceType=pbm.profile.ResourceType(resourceType='STORAGE'), profileCategory='REQUIREMENT')
    if len(profile_ids) > 0:
        storage_profiles = pm.PbmRetrieveContent(profileIds=profile_ids)
        for profile in storage_profiles:
            if profile.name == profile_name:
                storage_profile = profile
    else:
        self.module.warn('Unable to get storage profile IDs with STORAGE resource type and REQUIREMENT profile category.')
    return storage_profile