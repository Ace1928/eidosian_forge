from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware_spbm import SPBM
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec
def show_capabilities(self, capabilities):
    """
        Return property instance for given capabilities
        """
    capabilities_info = []
    for capability in capabilities:
        for constraint in capability.constraint:
            if hasattr(constraint, 'propertyInstance'):
                for propertyInstance in constraint.propertyInstance:
                    capabilities_info.append({'id': propertyInstance.id, 'value': propertyInstance.value})
    return capabilities_info