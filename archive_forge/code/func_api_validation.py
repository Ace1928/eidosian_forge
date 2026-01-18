from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.memset import get_zone_id
from ansible_collections.community.general.plugins.module_utils.memset import check_zone_domain
from ansible_collections.community.general.plugins.module_utils.memset import memset_api_call
def api_validation(args=None):
    """
    Perform some validation which will be enforced by Memset's API (see:
    https://www.memset.com/apidocs/methods_dns.html#dns.zone_domain_create)
    """
    if len(args['domain']) > 250:
        stderr = 'Zone domain must be less than 250 characters in length.'
        module.fail_json(failed=True, msg=stderr)