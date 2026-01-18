from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def get_time_in_seconds(time, time_units):
    """This method get time is seconds"""
    min_in_sec = 60
    hour_in_sec = 60 * 60
    day_in_sec = 24 * 60 * 60
    if time is not None and time > 0:
        if time_units in 'minutes':
            return time * min_in_sec
        elif time_units in 'hours':
            return time * hour_in_sec
        elif time_units in 'days':
            return time * day_in_sec
        else:
            return time
    else:
        return 0