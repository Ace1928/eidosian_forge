from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.memset import get_zone_id
from ansible_collections.community.general.plugins.module_utils.memset import check_zone_domain
from ansible_collections.community.general.plugins.module_utils.memset import memset_api_call
def delete_zone_domain(args=None, payload=None):
    """
    Deletion is pretty simple, domains are always unique so we
    we don't need to do any sanity checking to avoid deleting the
    wrong thing.
    """
    has_changed, has_failed = (False, False)
    msg, memset_api = (None, None)
    api_method = 'dns.zone_domain_list'
    _has_failed, _msg, response = memset_api_call(api_key=args['api_key'], api_method=api_method)
    domain_exists = check_zone_domain(data=response, domain=args['domain'])
    if domain_exists:
        api_method = 'dns.zone_domain_delete'
        payload['domain'] = args['domain']
        has_failed, msg, response = memset_api_call(api_key=args['api_key'], api_method=api_method, payload=payload)
        if not has_failed:
            has_changed = True
            memset_api = response.json()
            msg = None
    return (has_failed, has_changed, memset_api, msg)