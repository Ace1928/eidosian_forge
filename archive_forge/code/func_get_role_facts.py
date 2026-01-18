from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.text.converters import to_native
def get_role_facts(cursor, role=''):
    facts = {}
    cursor.execute("\n        select r.name, r.assigned_roles\n        from roles r\n        where (? = '' or r.name ilike ?)\n    ", role, role)
    while True:
        rows = cursor.fetchmany(100)
        if not rows:
            break
        for row in rows:
            role_key = row.name.lower()
            facts[role_key] = {'name': row.name, 'assigned_roles': []}
            if row.assigned_roles:
                facts[role_key]['assigned_roles'] = row.assigned_roles.replace(' ', '').split(',')
    return facts