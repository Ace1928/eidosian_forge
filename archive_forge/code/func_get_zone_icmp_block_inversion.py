from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils._text import to_native
from ansible_collections.ansible.posix.plugins.module_utils.version import StrictVersion
def get_zone_icmp_block_inversion(zone_settings):
    return zone_settings.getIcmpBlockInversion()