from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
def get_root_volume(self, key=None):
    args = {'account': self.get_account(key='name'), 'domainid': self.get_domain(key='id'), 'projectid': self.get_project(key='id'), 'virtualmachineid': self.get_vm(key='id'), 'type': 'ROOT'}
    volumes = self.query_api('listVolumes', **args)
    if volumes:
        return self._get_by_key(key, volumes['volume'][0])
    self.module.fail_json(msg="Root volume for '%s' not found" % self.get_vm('name'))