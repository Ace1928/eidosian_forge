from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import (
from ansible.module_utils._text import to_native
def sra_probe(self):
    if self.module.check_mode:
        self.changed = True
        return
    message = ''
    if self.support == 'remote' and (not self.is_remote_support_enabled()) or (self.support == 'onsite' and self.is_remote_support_enabled()):
        message += 'SRA configuration cannot be updated right now. '
    if any(self.add_proxy_details()):
        message += 'Proxy server details cannot be updated when SRA is enabled. '
    message += 'Please disable SRA and try to update.' if message else ''
    self.msg = message if message else self.msg
    return self.msg