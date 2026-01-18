from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware_spbm import SPBM
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec
from ansible_collections.community.vmware.plugins.module_utils.vmware_rest_client import VmwareRestClient
def create_mob_capability_constraint_instance(self, tag_id, tag_operator, tags):
    return pbm.capability.ConstraintInstance(propertyInstance=[self.create_mob_capability_property_instance(tag_id, tag_operator, tags)])