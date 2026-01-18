from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.manageiq import ManageIQ, manageiq_argument_spec
def compare_user(self, user, name, group_id, password, email):
    """ Compare user fields with new field values.

        Returns:
            false if user fields have some difference from new fields, true o/w.
        """
    found_difference = name and user['name'] != name or password is not None or (email and user['email'] != email) or (group_id and user['current_group_id'] != group_id)
    return not found_difference