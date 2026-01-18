from __future__ import absolute_import, division, print_function
import random
import sys
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule, create_multipart_formdata
from ansible.module_utils import six
from ansible.module_utils._text import to_native
def change_name_required(self):
    """Determine whether storage array name change is required."""
    if self.name is None:
        return False
    current_configuration = self.get_current_configuration()
    if self.name and len(self.name) > 30:
        self.module.fail_json(msg='The provided name is invalid, it must be less than or equal to 30 characters in length. Array [%s]' % self.ssid)
    return self.name != current_configuration['name']