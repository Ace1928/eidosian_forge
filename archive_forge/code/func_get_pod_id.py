from __future__ import absolute_import, division, print_function
import base64
from ansible.module_utils._text import to_bytes, to_text
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
def get_pod_id(self):
    pod_name = self.module.params.get('pod')
    if not pod_name:
        return None
    args = {'zoneid': self.get_zone(key='id')}
    pods = self.query_api('listPods', **args)
    if pods:
        for p in pods['pod']:
            if pod_name in [p['name'], p['id']]:
                return p['id']
    self.fail_json(msg="Pod '%s' not found" % pod_name)