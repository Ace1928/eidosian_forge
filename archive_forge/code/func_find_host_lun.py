from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible_collections.infinidat.infinibox.plugins.module_utils.infinibox import (
def find_host_lun(host, volume):
    """ Find a hosts lun """
    found_lun = None
    luns = host.get_luns()
    for lun in luns:
        if lun.volume == volume:
            found_lun = lun.lun
    return found_lun