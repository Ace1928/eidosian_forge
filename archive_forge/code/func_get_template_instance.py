from __future__ import (absolute_import, division, print_function)
from ansible_collections.community.general.plugins.module_utils.opennebula import OpenNebulaModule
def get_template_instance(self, requested_id, requested_name):
    if requested_id:
        return self.get_template_by_id(requested_id)
    else:
        return self.get_template_by_name(requested_name)