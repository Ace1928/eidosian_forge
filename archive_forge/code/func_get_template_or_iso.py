from __future__ import absolute_import, division, print_function
import base64
from ansible.module_utils._text import to_bytes, to_text
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
def get_template_or_iso(self, key=None):
    template = self.module.params.get('template')
    iso = self.module.params.get('iso')
    if not template and (not iso):
        return None
    args = {'account': self.get_account(key='name'), 'domainid': self.get_domain(key='id'), 'projectid': self.get_project(key='id'), 'zoneid': self.get_zone(key='id'), 'isrecursive': True, 'fetch_list': True}
    if template:
        if self.template:
            return self._get_by_key(key, self.template)
        rootdisksize = self.module.params.get('root_disk_size')
        args['templatefilter'] = self.module.params.get('template_filter')
        args['fetch_list'] = True
        templates = self.query_api('listTemplates', **args)
        if templates:
            for t in templates:
                if template in [t.get('displaytext', None), t['name'], t['id']]:
                    if rootdisksize and t['size'] > rootdisksize * 1024 ** 3:
                        continue
                    self.template = t
                    return self._get_by_key(key, self.template)
        if rootdisksize:
            more_info = ' (with size <= %s)' % rootdisksize
        else:
            more_info = ''
        self.module.fail_json(msg="Template '%s' not found%s" % (template, more_info))
    elif iso:
        if self.iso:
            return self._get_by_key(key, self.iso)
        args['isofilter'] = self.module.params.get('template_filter')
        args['fetch_list'] = True
        isos = self.query_api('listIsos', **args)
        if isos:
            for i in isos:
                if iso in [i['displaytext'], i['name'], i['id']]:
                    self.iso = i
                    return self._get_by_key(key, self.iso)
        self.module.fail_json(msg="ISO '%s' not found" % iso)