from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils.api import basic_auth_argument_spec
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.error import HTTPError
@property
def snapshot_group_id(self):
    url = self.url + 'storage-systems/%s/snapshot-groups' % self.ssid
    try:
        rc, data = request(url, headers=HEADERS, url_username=self.user, url_password=self.pwd, validate_certs=self.certs)
    except Exception as err:
        self.module.fail_json(msg='Failed to fetch snapshot groups. ' + 'Id [%s]. Error [%s].' % (self.ssid, to_native(err)))
    for ssg in data:
        if ssg['name'] == self.name:
            self.ssg_data = ssg
            return ssg['id']
    return None