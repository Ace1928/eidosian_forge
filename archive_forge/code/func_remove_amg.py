from __future__ import absolute_import, division, print_function
import json
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.netapp import request, eseries_host_argument_spec
def remove_amg(module, ssid, api_url, pwd, user, async_id):
    endpoint = 'storage-systems/%s/async-mirrors/%s' % (ssid, async_id)
    url = api_url + endpoint
    try:
        rc, data = request(url, method='DELETE', url_username=user, url_password=pwd, headers=HEADERS)
    except Exception as e:
        module.exit_json(msg='Exception while removing async mirror group. Message: %s' % to_native(e), exception=traceback.format_exc())
    return