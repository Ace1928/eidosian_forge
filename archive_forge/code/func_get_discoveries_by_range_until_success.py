from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (
import time
import re
def get_discoveries_by_range_until_success(self):
    """
        Continuously retrieve a specific discovery by name from a range of
        discoveries in the Cisco Catalyst Center until the discovery is complete.

        Returns:
          - discovery: The completed discovery with the specified name from
                       the range of discoveries. If the discovery is not
                       found or not completed, the function fails the module
                       and returns None.
        """
    result = False
    discovery = self.lookup_discovery_by_range_via_name()
    if not discovery:
        msg = 'Cannot find any discovery task with name {0} -- Discovery result: {1}'.format(str(self.validated_config[0].get('discovery_name')), str(discovery))
        self.log(msg, 'INFO')
        self.module.fail_json(msg=msg)
    while True:
        discovery = self.lookup_discovery_by_range_via_name()
        if discovery.get('discoveryCondition') == 'Complete':
            result = True
            break
        time.sleep(3)
    if not result:
        msg = 'Cannot find any discovery task with name {0} -- Discovery result: {1}'.format(str(self.validated_config[0].get('discovery_name')), str(discovery))
        self.log(msg, 'CRITICAL')
        self.module.fail_json(msg=msg)
    self.result.update(dict(discovery_range=discovery))
    return discovery