from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.memset import get_zone_id
from ansible_collections.community.general.plugins.module_utils.memset import memset_api_call
def delete_zone_record(args=None, records=None, payload=None):
    """
    Matching records can be cleanly deleted without affecting other
    resource types, so this is pretty simple to achieve.
    """
    has_changed, has_failed = (False, False)
    msg, memset_api = (None, None)
    if records:
        for zone_record in records:
            if args['check_mode']:
                has_changed = True
                return (has_changed, has_failed, memset_api, msg)
            payload['id'] = zone_record['id']
            api_method = 'dns.zone_record_delete'
            has_failed, msg, response = memset_api_call(api_key=args['api_key'], api_method=api_method, payload=payload)
            if not has_failed:
                has_changed = True
                memset_api = zone_record
                msg = None
    return (has_changed, has_failed, memset_api, msg)