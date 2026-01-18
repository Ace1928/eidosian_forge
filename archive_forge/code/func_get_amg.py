from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils.api import basic_auth_argument_spec
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.urls import open_url
def get_amg(self):
    endpoint = self.url + '/storage-systems/%s/async-mirrors' % self.ssid
    rc, amg_objs = request(endpoint, url_username=self.user, url_password=self.pwd, validate_certs=self.certs, headers=self.post_headers)
    try:
        amg_id = filter(lambda d: d['label'] == self.name, amg_objs)[0]['id']
        amg_obj = filter(lambda d: d['label'] == self.name, amg_objs)[0]
    except IndexError:
        self.module.fail_json(msg='There is no async mirror group  %s associated with storage array %s' % (self.name, self.ssid))
    return (amg_id, amg_obj)