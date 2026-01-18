from __future__ import absolute_import, division, print_function
import base64
from ansible.module_utils._text import to_bytes, to_text
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
def restore_instance(self):
    instance = self.get_instance()
    self.result['changed'] = True
    if instance:
        args = {}
        args['templateid'] = self.get_template_or_iso(key='id')
        args['virtualmachineid'] = instance['id']
        res = self.query_api('restoreVirtualMachine', **args)
        poll_async = self.module.params.get('poll_async')
        if poll_async:
            instance = self.poll_job(res, 'virtualmachine')
    return instance