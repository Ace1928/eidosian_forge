from __future__ import absolute_import, division, print_function
import json
import os
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.aci.plugins.module_utils.aci import ACIModule, aci_argument_spec
from ansible.module_utils._text import to_text
def response_type(self, rawoutput, rest_type='xml'):
    """Handle APIC response output"""
    if rest_type == 'json':
        self.response_json(rawoutput)
    else:
        self.response_xml(rawoutput)
    if HAS_URLPARSE:
        self.result['changed'] = self.changed(self.imdata)