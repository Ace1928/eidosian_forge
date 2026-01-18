from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware_spbm import SPBM
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec
from ansible_collections.community.vmware.plugins.module_utils.vmware_rest_client import VmwareRestClient
def get_tag_constraints(self, capabilities):
    """
        Return tag constraints for a profile given its capabilities
        """
    tag_constraints = {}
    for capability in capabilities:
        for constraint in capability.constraint:
            if hasattr(constraint, 'propertyInstance'):
                for propertyInstance in constraint.propertyInstance:
                    if hasattr(propertyInstance.value, 'values'):
                        tag_constraints['id'] = propertyInstance.id
                        tag_constraints['values'] = propertyInstance.value.values
                        tag_constraints['operator'] = propertyInstance.operator
    return tag_constraints