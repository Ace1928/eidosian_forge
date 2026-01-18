from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import (
from ansible.module_utils._text import to_native
def portset_probe(self):
    updates = []
    if self.portset_type and self.portset_type != self.portset_details['type']:
        self.module.fail_json(msg="portset_type can't be updated for portset")
    if self.porttype and self.porttype != self.portset_details['port_type']:
        self.module.fail_json(msg="porttype can't be updated for portset")
    if self.ownershipgroup and self.ownershipgroup != self.portset_details['owner_name']:
        updates.append('ownershipgroup')
    if self.noownershipgroup:
        updates.append('noownershipgroup')
    self.log('Modifications to be done: %s', updates)
    return updates