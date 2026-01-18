from __future__ import absolute_import, division, print_function
import re
from copy import deepcopy
def cr_key_lookup(key, mo):
    """
    Helper method to get instance key value for Managed Object (mo)
    """
    cr_keys = [key]
    if key == 'destination_groups' and mo == 'TMS_DESTGROUP':
        cr_keys = ['destination']
    elif key == 'sensor_groups' and mo == 'TMS_SENSORGROUP':
        cr_keys = ['data_source', 'path']
    elif key == 'subscriptions' and mo == 'TMS_SUBSCRIPTION':
        cr_keys = ['destination_group', 'sensor_group']
    return cr_keys