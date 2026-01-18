from __future__ import (absolute_import, division, print_function)
import string
import json
import re
from ansible.module_utils.six import iteritems
from ansible_collections.community.mysql.plugins.module_utils.mysql import (
def limit_resources(module, cursor, user, host, resource_limits, check_mode):
    """Limit user resources.

    Args:
        module (AnsibleModule): Ansible module object.
        cursor (cursor): DB driver cursor object.
        user (str): User name.
        host (str): User host name.
        resource_limit (dict): Dictionary with desired limits.
        check_mode (bool): Run the function in check mode or not.

    Returns: True, if changed, False otherwise.
    """
    if not impl.server_supports_alter_user(cursor):
        module.fail_json(msg="The server version does not match the requirements for resource_limits parameter. See module's documentation.")
    cursor.execute('SELECT VERSION()')
    if 'mariadb' not in cursor.fetchone()[0].lower():
        if 'MAX_STATEMENT_TIME' in resource_limits:
            module.fail_json(msg='MAX_STATEMENT_TIME resource limit is only supported by MariaDB.')
    current_limits = get_resource_limits(cursor, user, host)
    needs_to_change = match_resource_limits(module, current_limits, resource_limits)
    if not needs_to_change:
        return False
    if needs_to_change and check_mode:
        return True
    tmp = []
    for key, val in iteritems(needs_to_change):
        tmp.append('%s %s' % (key, val))
    query = 'ALTER USER %s@%s'
    query += ' WITH %s' % ' '.join(tmp)
    cursor.execute(query, (user, host))
    return True