from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.mongodb.plugins.module_utils.mongodb_common import (
def balancing_window(client, start, stop):
    s = False
    if start is not None and stop is not None:
        result = client['config'].settings.find_one({'_id': 'balancer', 'activeWindow.start': start, 'activeWindow.stop': stop})
    else:
        result = client['config'].settings.find_one({'_id': 'balancer', 'activeWindow': {'$exists': True}})
    if result:
        s = True
    return s