from __future__ import absolute_import, division, print_function
import xml.etree.ElementTree as ET
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import fetch_url
def get_logbundle(self):
    self.validate_manifests()
    url = 'https://' + self.esxi_hostname + '/cgi-bin/vm-support.cgi?manifests=' + '&'.join(self.manifests)
    if self.performance_data:
        duration = self.performance_data.get('duration')
        interval = self.performance_data.get('interval')
        url = url + '&performance=true&duration=%s&interval=%s' % (duration, interval)
    headers = self.generate_req_headers(url)
    try:
        resp, info = fetch_url(self.module, method='GET', headers=headers, url=url)
        if info['status'] != 200:
            self.module.fail_json(msg='failed to fetch logbundle from %s: %s' % (url, info['msg']))
        with open(self.dest, 'wb') as local_file:
            local_file.write(resp.read())
    except Exception as e:
        self.module.fail_json(msg='Failed to fetch logbundle from %s: %s' % (url, e))
    self.module.exit_json(changed=True, dest=self.dest)