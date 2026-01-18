from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import fetch_url
def get_geo_data(self):
    response, info = fetch_url(self.module, self.url, force=True, timeout=self.timeout)
    try:
        info['status'] == 200
    except AssertionError:
        self.module.fail_json(msg='Could not get {0} page, check for connectivity!'.format(self.url))
    else:
        try:
            content = response.read()
            result = self.module.from_json(content.decode('utf8'))
        except ValueError:
            self.module.fail_json(msg='Failed to parse the ipinfo.io response: {0} {1}'.format(self.url, content))
        else:
            return result