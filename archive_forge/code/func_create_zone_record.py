from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.memset import get_zone_id
from ansible_collections.community.general.plugins.module_utils.memset import memset_api_call
def create_zone_record(args=None, zone_id=None, records=None, payload=None):
    """
    Sanity checking has already occurred prior to this function being
    called, so we can go ahead and either create or update the record.
    As defaults are defined for all values in the argument_spec, this
    may cause some changes to occur as the defaults are enforced (if
    the user has only configured required variables).
    """
    has_changed, has_failed = (False, False)
    msg, memset_api = (None, None)
    new_record = dict()
    new_record['zone_id'] = zone_id
    for arg in ['priority', 'address', 'relative', 'record', 'ttl', 'type']:
        new_record[arg] = args[arg]
    if records:
        for zone_record in records:
            new_record['id'] = zone_record['id']
            if zone_record == new_record:
                memset_api = zone_record
                return (has_changed, has_failed, memset_api, msg)
            else:
                payload = zone_record.copy()
                payload.update(new_record)
                api_method = 'dns.zone_record_update'
                if args['check_mode']:
                    has_changed = True
                    memset_api = new_record
                    return (has_changed, has_failed, memset_api, msg)
                has_failed, msg, response = memset_api_call(api_key=args['api_key'], api_method=api_method, payload=payload)
                if not has_failed:
                    has_changed = True
                    memset_api = new_record
                    msg = None
    else:
        api_method = 'dns.zone_record_create'
        payload = new_record
        if args['check_mode']:
            has_changed = True
            memset_api = new_record
            return (has_changed, has_failed, memset_api, msg)
        has_failed, msg, response = memset_api_call(api_key=args['api_key'], api_method=api_method, payload=payload)
        if not has_failed:
            has_changed = True
            memset_api = new_record
            msg = None
    return (has_changed, has_failed, memset_api, msg)