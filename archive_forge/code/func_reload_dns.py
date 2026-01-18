from __future__ import (absolute_import, division, print_function)
from time import sleep
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.memset import memset_api_call
def reload_dns(args=None):
    """
    DNS reloads are a single API call and therefore there's not much
    which can go wrong outside of auth errors.
    """
    retvals, payload = (dict(), dict())
    has_changed, has_failed = (False, False)
    memset_api, msg, stderr = (None, None, None)
    api_method = 'dns.reload'
    has_failed, msg, response = memset_api_call(api_key=args['api_key'], api_method=api_method)
    if has_failed:
        retvals['failed'] = has_failed
        if response.status_code is not None:
            retvals['memset_api'] = response.json()
        else:
            retvals['stderr'] = response.stderr
        retvals['msg'] = msg
        return retvals
    has_changed = True
    memset_api = msg
    msg = None
    if args['poll']:
        job_id = response.json()['id']
        memset_api, msg, stderr = poll_reload_status(api_key=args['api_key'], job_id=job_id, payload=payload)
    retvals['failed'] = has_failed
    retvals['changed'] = has_changed
    for val in ['msg', 'stderr', 'memset_api']:
        if val is not None:
            retvals[val] = eval(val)
    return retvals