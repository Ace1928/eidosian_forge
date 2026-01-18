from __future__ import absolute_import, division, print_function
import re
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.six import iteritems
def get_member_status(self):
    """ Returns a dictionary of a balancer member's status attributes."""
    status_mapping = {'disabled': 'Dis', 'drained': 'Drn', 'hot_standby': 'Stby', 'ignore_errors': 'Ign'}
    actual_status = str(self.attributes['Status'])
    status = dict(((mode, patt in actual_status) for mode, patt in iteritems(status_mapping)))
    return status