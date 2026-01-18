from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def get_schedule_value(type):
    """Get the enum for schedule.
            :param type: The type of rule
            :return: The enum value for rule
    """
    rule_type = {'every_n_hours': 0, 'every_day': 1, 'every_n_days': 2, 'every_week': 3, 'every_month': 4}
    return rule_type.get(type)