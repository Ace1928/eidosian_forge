from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils.api import basic_auth_argument_spec
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.error import HTTPError
def oldest_image(module, ssid, api_url, api_pwd, api_usr, name):
    get_status = 'storage-systems/%s/snapshot-images' % ssid
    url = api_url + get_status
    try:
        ret, images = request(url, url_username=api_usr, url_password=api_pwd, headers=HEADERS, validate_certs=module.params['validate_certs'])
    except Exception as err:
        module.fail_json(msg='Failed to get snapshot images for group. Group [%s]. Id [%s]. Error [%s]' % (name, ssid, to_native(err)))
    if not images:
        module.exit_json(msg='There are no snapshot images to remove.  Group [%s]. Id [%s].' % (name, ssid))
    oldest = min(images, key=lambda x: x['pitSequenceNumber'])
    if oldest is None or 'pitRef' not in oldest:
        module.fail_json(msg='Failed to lookup oldest snapshot group.  Group [%s]. Id [%s].' % (name, ssid))
    return oldest