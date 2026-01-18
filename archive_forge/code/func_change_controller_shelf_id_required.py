from __future__ import absolute_import, division, print_function
import random
import sys
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule, create_multipart_formdata
from ansible.module_utils import six
from ansible.module_utils._text import to_native
def change_controller_shelf_id_required(self):
    """Determine whether storage array tray identifier change is required."""
    current_configuration = self.get_current_configuration()
    if self.controller_shelf_id is not None and self.controller_shelf_id != current_configuration['controller_shelf_id']:
        if self.controller_shelf_id in current_configuration['used_shelf_ids']:
            self.module.fail_json(msg='The controller_shelf_id is currently being used by another shelf. Used Identifiers: [%s]. Array [%s].' % (', '.join([str(id) for id in self.get_current_configuration()['used_shelf_ids']]), self.ssid))
        if self.controller_shelf_id < 0 or self.controller_shelf_id > self.LAST_AVAILABLE_CONTROLLER_SHELF_ID:
            self.module.fail_json(msg='The controller_shelf_id must be 0-99 and not already used by another shelf. Used Identifiers: [%s]. Array [%s].' % (', '.join([str(id) for id in self.get_current_configuration()['used_shelf_ids']]), self.ssid))
        return True
    return False