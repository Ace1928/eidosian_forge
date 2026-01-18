from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def get_days_of_week_enum(self, days_of_week):
    """Get the enum for days of week.
            :param days_of_week: The list of days of week
            :return: The list of days_of_week enum
        """
    days_of_week_enum = []
    for day in days_of_week:
        if day in utils.DayOfWeekEnum.__members__:
            days_of_week_enum.append(utils.DayOfWeekEnum[day])
        else:
            errormsg = 'Invalid choice {0} for days of week'.format(day)
            LOG.error(errormsg)
            self.module.fail_json(msg=errormsg)
    return days_of_week_enum