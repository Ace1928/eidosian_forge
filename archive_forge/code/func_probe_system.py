from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import IBMSVCRestApi, svc_argument_spec, get_logger
from ansible.module_utils._text import to_native
import time
def probe_system(self, data):
    modify = {}
    if self.invemailinterval:
        if self.invemailinterval != data['inventory_mail_interval']:
            modify['invemailinterval'] = self.invemailinterval
    if self.enhancedcallhome:
        if self.enhancedcallhome != data['enhanced_callhome']:
            modify['enhancedcallhome'] = self.enhancedcallhome
    if self.censorcallhome:
        if self.censorcallhome != data['enhanced_callhome']:
            modify['censorcallhome'] = self.censorcallhome
    return modify