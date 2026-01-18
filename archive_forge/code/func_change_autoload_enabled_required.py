from __future__ import absolute_import, division, print_function
import random
import sys
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule, create_multipart_formdata
from ansible.module_utils import six
from ansible.module_utils._text import to_native
def change_autoload_enabled_required(self):
    """Determine whether automatic load balancing state change is required."""
    if self.autoload_enabled is None:
        return False
    change_required = False
    current_configuration = self.get_current_configuration()
    if self.autoload_enabled and (not current_configuration['autoload_capable']):
        self.module.fail_json(msg='Automatic load balancing is not available. Array [%s].' % self.ssid)
    if self.autoload_enabled:
        if not current_configuration['autoload_enabled'] or not current_configuration['host_connectivity_reporting_enabled']:
            change_required = True
    elif current_configuration['autoload_enabled']:
        change_required = True
    return change_required