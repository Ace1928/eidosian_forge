from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.memset import get_zone_id
from ansible_collections.community.general.plugins.module_utils.memset import check_zone_domain
from ansible_collections.community.general.plugins.module_utils.memset import memset_api_call
def create_or_delete_domain(args=None):
    """
    We need to perform some initial sanity checking and also look
    up required info before handing it off to create or delete.
    """
    retvals, payload = (dict(), dict())
    has_changed, has_failed = (False, False)
    msg, stderr, memset_api = (None, None, None)
    api_method = 'dns.zone_list'
    has_failed, msg, response = memset_api_call(api_key=args['api_key'], api_method=api_method)
    if has_failed:
        retvals['failed'] = has_failed
        retvals['msg'] = msg
        if response.status_code is not None:
            retvals['stderr'] = 'API returned an error: {0}'.format(response.status_code)
        else:
            retvals['stderr'] = response.stderr
        return retvals
    zone_exists, msg, counter, zone_id = get_zone_id(zone_name=args['zone'], current_zones=response.json())
    if not zone_exists:
        has_failed = True
        if counter == 0:
            stderr = "DNS zone '{0}' does not exist, cannot create domain.".format(args['zone'])
        elif counter > 1:
            stderr = '{0} matches multiple zones, cannot create domain.'.format(args['zone'])
        retvals['failed'] = has_failed
        retvals['msg'] = stderr
        return retvals
    if args['state'] == 'present':
        has_failed, has_changed, msg = create_zone_domain(args=args, zone_exists=zone_exists, zone_id=zone_id, payload=payload)
    if args['state'] == 'absent':
        has_failed, has_changed, memset_api, msg = delete_zone_domain(args=args, payload=payload)
    retvals['changed'] = has_changed
    retvals['failed'] = has_failed
    for val in ['msg', 'stderr', 'memset_api']:
        if val is not None:
            retvals[val] = eval(val)
    return retvals