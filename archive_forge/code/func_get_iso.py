from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
def get_iso(self):
    if not self.iso:
        args = {'isready': self.module.params.get('is_ready'), 'isofilter': self.module.params.get('iso_filter'), 'domainid': self.get_domain('id'), 'account': self.get_account('name'), 'projectid': self.get_project('id')}
        if not self.module.params.get('cross_zones'):
            args['zoneid'] = self.get_zone(key='id')
        checksum = self.module.params.get('checksum')
        if not checksum:
            args['name'] = self.module.params.get('name')
        isos = self.query_api('listIsos', **args)
        if isos:
            if not checksum:
                self.iso = isos['iso'][0]
            else:
                for i in isos['iso']:
                    if i['checksum'] == checksum:
                        self.iso = i
                        break
    return self.iso