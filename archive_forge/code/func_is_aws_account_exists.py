from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import (
from ansible.module_utils._text import to_native
def is_aws_account_exists(self, name=None):
    result = {}
    cmd = 'lscloudaccount'
    name = name if name else self.name
    data = self.restapi.svc_obj_info(cmd=cmd, cmdopts=None, cmdargs=[name])
    if isinstance(data, list):
        for d in data:
            result.update(d)
    else:
        result = data
    self.aws_data = result
    return result