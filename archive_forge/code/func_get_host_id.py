from __future__ import absolute_import, division, print_function
import base64
from ansible.module_utils._text import to_bytes, to_text
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
def get_host_id(self):
    host_name = self.module.params.get('host')
    if not host_name:
        return None
    args = {'type': 'routing', 'zoneid': self.get_zone(key='id')}
    hosts = self.query_api('listHosts', **args)
    if hosts:
        for h in hosts['host']:
            if host_name in [h['name'], h['id']]:
                return h['id']
    self.fail_json(msg="Host '%s' not found" % host_name)