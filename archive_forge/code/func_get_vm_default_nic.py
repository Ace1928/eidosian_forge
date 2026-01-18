from __future__ import absolute_import, division, print_function
import os
import sys
import time
import traceback
from ansible.module_utils._text import to_text, to_native
from ansible.module_utils.basic import missing_required_lib, env_fallback
def get_vm_default_nic(self):
    if self.vm_default_nic:
        return self.vm_default_nic
    nics = self.query_api('listNics', virtualmachineid=self.get_vm(key='id'))
    if nics:
        for n in nics['nic']:
            if n['isdefault']:
                self.vm_default_nic = n
                return self.vm_default_nic
    self.fail_json(msg="No default IP address of VM '%s' found" % self.module.params.get('vm'))