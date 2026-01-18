from __future__ import absolute_import, division, print_function
import os
import traceback
import xml.etree.ElementTree as ET
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.text.converters import to_native
def switch_state(self):
    if self.enabled is False:
        self.server.disable_job(self.name)
    else:
        self.server.enable_job(self.name)