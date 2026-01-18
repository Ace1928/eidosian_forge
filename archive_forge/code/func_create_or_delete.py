from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.memset import check_zone
from ansible_collections.community.general.plugins.module_utils.memset import get_zone_id
from ansible_collections.community.general.plugins.module_utils.memset import memset_api_call
def create_or_delete(args=None):
    """
    We need to perform some initial sanity checking and also look
    up required info before handing it off to create or delete.
    """
    retvals, payload = (dict(), dict())
    has_failed, has_changed = (False, False)
    msg, memset_api, stderr = (None, None, None)
    api_method = 'dns.zone_list'
    _has_failed, _msg, response = memset_api_call(api_key=args['api_key'], api_method=api_method)
    if _has_failed:
        retvals['failed'] = _has_failed
        retvals['msg'] = _msg
        if response.stderr is not None:
            retvals['stderr'] = response.stderr
        return retvals
    zone_exists, _msg, counter, _zone_id = get_zone_id(zone_name=args['name'], current_zones=response.json())
    if args['state'] == 'present':
        has_failed, has_changed, memset_api, msg = create_zone(args=args, zone_exists=zone_exists, payload=payload)
    elif args['state'] == 'absent':
        has_failed, has_changed, memset_api, msg = delete_zone(args=args, zone_exists=zone_exists, payload=payload)
    retvals['failed'] = has_failed
    retvals['changed'] = has_changed
    for val in ['msg', 'stderr', 'memset_api']:
        if val is not None:
            retvals[val] = eval(val)
    return retvals