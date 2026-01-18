from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.mongodb.plugins.module_utils.mongodb_common import (
def stop_balancer(client):
    """
    Stops MongoDB balancer
    """
    cmd_doc = OrderedDict([('balancerStop', 1), ('maxTimeMS', 60000)])
    client['admin'].command(cmd_doc)
    time.sleep(1)