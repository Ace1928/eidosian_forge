from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.memset import get_zone_id
from ansible_collections.community.general.plugins.module_utils.memset import check_zone_domain
from ansible_collections.community.general.plugins.module_utils.memset import memset_api_call
def create_zone_domain(args=None, zone_exists=None, zone_id=None, payload=None):
    """
    At this point we already know whether the containing zone exists,
    so we just need to create the domain (or exit if it already exists).
    """
    has_changed, has_failed = (False, False)
    msg = None
    api_method = 'dns.zone_domain_list'
    _has_failed, _msg, response = memset_api_call(api_key=args['api_key'], api_method=api_method)
    for zone_domain in response.json():
        if zone_domain['domain'] == args['domain']:
            has_changed = False
            break
    else:
        api_method = 'dns.zone_domain_create'
        payload['domain'] = args['domain']
        payload['zone_id'] = zone_id
        has_failed, msg, response = memset_api_call(api_key=args['api_key'], api_method=api_method, payload=payload)
        if not has_failed:
            has_changed = True
    return (has_failed, has_changed, msg)