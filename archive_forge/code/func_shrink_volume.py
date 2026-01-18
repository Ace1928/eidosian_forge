from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import (
from ansible.module_utils._text import to_native
def shrink_volume(self, shrink_size):
    self.restapi.svc_run_command('shrinkvdisksize', {'size': shrink_size, 'unit': 'b'}, [self.name])
    self.changed = True