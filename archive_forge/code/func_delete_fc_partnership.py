from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import (
from ansible.module_utils._text import to_native
def delete_fc_partnership(self, restapi, cluster):
    if self.module.check_mode:
        self.changed = True
        return
    restapi.svc_run_command('rmpartnership', None, [cluster])
    self.changed = True