from __future__ import (absolute_import, division, print_function)
from ansible.plugins.lookup import LookupBase
from ansible.errors import AnsibleError
def change_volume_mapping_lun(self, name, host, lun):
    """remove volume mapping to record table (luns_by_target)."""
    self.remove_volume_mapping(name, host)
    self.add_volume_mapping(name, host, lun)