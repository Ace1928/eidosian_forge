from __future__ import absolute_import, division, print_function
import datetime
import time
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.scaleway import SCALEWAY_LOCATION, scaleway_argument_spec, Scaleway
def public_ip_payload(compute_api, public_ip):
    if public_ip in ('absent',):
        return {'dynamic_ip_required': False}
    if public_ip in ('dynamic', 'allocated'):
        return {'dynamic_ip_required': True}
    response = compute_api.get('ips')
    if not response.ok:
        msg = 'Error during public IP validation: (%s) %s' % (response.status_code, response.json)
        compute_api.module.fail_json(msg=msg)
    ip_list = []
    try:
        ip_list = response.json['ips']
    except KeyError:
        compute_api.module.fail_json(msg='Error in getting the IP information from: %s' % response.json)
    lookup = [ip['id'] for ip in ip_list]
    if public_ip in lookup:
        return {'public_ip': public_ip}