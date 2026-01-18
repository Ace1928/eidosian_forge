from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.manageiq import ManageIQ, manageiq_argument_spec
def class_name_to_type(self, class_name):
    """ Convert class_name to type

        Returns:
            the type
        """
    out = [k for k, v in supported_providers().items() if v['class_name'] == class_name]
    if len(out) == 1:
        return out[0]
    return None