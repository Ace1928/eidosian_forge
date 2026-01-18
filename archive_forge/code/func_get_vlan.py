from __future__ import (absolute_import, division, print_function)
import os
import re
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib  # noqa: F401, pylint: disable=unused-import
from ansible.module_utils.six.moves import configparser
from os.path import expanduser
from uuid import UUID
def get_vlan(self, locator, location, network_domain):
    """
        Get a VLAN object by its name or id
        """
    if is_uuid(locator):
        vlan = self.driver.ex_get_vlan(locator)
    else:
        matching_vlans = [vlan for vlan in self.driver.ex_list_vlans(location, network_domain) if vlan.name == locator]
        if matching_vlans:
            vlan = matching_vlans[0]
        else:
            vlan = None
    if vlan:
        return vlan
    raise UnknownVLANError("VLAN '%s' could not be found" % locator)