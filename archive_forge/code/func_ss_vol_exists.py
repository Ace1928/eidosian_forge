from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils.api import basic_auth_argument_spec
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.error import HTTPError
@property
def ss_vol_exists(self):
    rc, ss_vols = request(self.url + 'storage-systems/%s/snapshot-volumes' % self.ssid, headers=HEADERS, url_username=self.user, url_password=self.pwd, validate_certs=self.certs)
    if ss_vols:
        for ss_vol in ss_vols:
            if ss_vol['name'] == self.name:
                self.ss_vol = ss_vol
                return True
    else:
        return False
    return False