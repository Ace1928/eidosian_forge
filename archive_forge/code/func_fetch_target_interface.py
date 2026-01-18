from __future__ import absolute_import, division, print_function
import json
import logging
from pprint import pformat
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.netapp import request, eseries_host_argument_spec
from ansible.module_utils._text import to_native
def fetch_target_interface(self):
    interfaces = self.interfaces
    for iface in interfaces:
        if iface['channel'] == self.name and self.controllers[self.controller] == iface['controllerId']:
            return iface
    channels = sorted(set((str(iface['channel']) for iface in interfaces if self.controllers[self.controller] == iface['controllerId'])))
    self.module.fail_json(msg='The requested channel of %s is not valid. Valid channels include: %s.' % (self.name, ', '.join(channels)))