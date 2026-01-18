from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils.six.moves.urllib.parse import urlencode
from ansible_collections.community.hrobot.plugins.module_utils.robot import (
def get_failover_state(value):
    """
    Create result dictionary for failover IP's value.

    The value ``None`` represents unrouted.
    """
    return dict(value=value, state='routed' if value else 'unrouted')