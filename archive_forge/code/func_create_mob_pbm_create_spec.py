from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware_spbm import SPBM
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec
from ansible_collections.community.vmware.plugins.module_utils.vmware_rest_client import VmwareRestClient
def create_mob_pbm_create_spec(self, tag_id, tag_operator, tags, tag_category, description, name):
    return pbm.profile.CapabilityBasedProfileCreateSpec(name=name, description=description, resourceType=pbm.profile.ResourceType(resourceType='STORAGE'), category='REQUIREMENT', constraints=self.create_mob_capability_subprofile(tag_id, tag_operator, tags, tag_category))