from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.oneandone import (
def delete_public_ip(module, oneandone_conn):
    """
    Delete a public IP

    module : AnsibleModule object
    oneandone_conn: authenticated oneandone object

    Returns a dictionary containing a 'changed' attribute indicating whether
    any public IP was deleted.
    """
    public_ip_id = module.params.get('public_ip_id')
    public_ip = get_public_ip(oneandone_conn, public_ip_id, True)
    if public_ip is None:
        _check_mode(module, False)
        module.fail_json(msg='public IP %s not found.' % public_ip_id)
    try:
        _check_mode(module, True)
        deleted_public_ip = oneandone_conn.delete_public_ip(ip_id=public_ip['id'])
        changed = True if deleted_public_ip else False
        return (changed, {'id': public_ip['id']})
    except Exception as e:
        module.fail_json(msg=str(e))