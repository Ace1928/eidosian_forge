from __future__ import absolute_import, division, print_function
import json
import logging
import sys
import traceback
from ansible.module_utils.api import basic_auth_argument_spec
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves import reduce
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils._text import to_native
from ansible.module_utils.urls import open_url
def get_candidate_disks(self, disk_count, size_unit='gb', capacity=None):
    self.debug('getting candidate disks...')
    drives_req = dict(driveCount=disk_count, sizeUnit=size_unit, driveType='ssd')
    if capacity:
        drives_req['targetUsableCapacity'] = capacity
    rc, drives_resp = request(self.api_url + '/storage-systems/%s/drives' % self.ssid, data=json.dumps(drives_req), headers=self.post_headers, method='POST', url_username=self.api_username, url_password=self.api_password, validate_certs=self.validate_certs)
    if rc == 204:
        self.module.fail_json(msg='Cannot find disks to match requested criteria for ssd cache')
    disk_ids = [d['id'] for d in drives_resp]
    bytes = reduce(lambda s, d: s + int(d['usableCapacity']), drives_resp, 0)
    return (disk_ids, bytes)