from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (
import time
import re
def get_exist_discovery(self):
    """
        Retrieve an existing discovery by its name from a range of discoveries.

        Returns:
          - discovery: The discovery with the specified name from the range of
                       discoveries. If no matching discovery is found, it
                       returns None and updates the 'exist_discovery' entry in
                       the result dictionary to None.
        """
    discovery = self.lookup_discovery_by_range_via_name()
    if not discovery:
        self.result.update(dict(exist_discovery=discovery))
        return None
    have = dict(exist_discovery=discovery)
    self.have = have
    self.result.update(dict(exist_discovery=discovery))
    return discovery