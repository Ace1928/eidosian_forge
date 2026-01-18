from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.manageiq import ManageIQ, manageiq_argument_spec
@staticmethod
def normalized_managed_tag_filters_to_miq(norm_managed_filters):
    if not norm_managed_filters:
        return None
    return list(norm_managed_filters.values())