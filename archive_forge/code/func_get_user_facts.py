from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.text.converters import to_native
def get_user_facts(cursor, user=''):
    facts = {}
    cursor.execute("\n        select u.user_name, u.is_locked, u.lock_time,\n        p.password, p.acctexpired as is_expired,\n        u.profile_name, u.resource_pool,\n        u.all_roles, u.default_roles\n        from users u join password_auditor p on p.user_id = u.user_id\n        where not u.is_super_user\n        and (? = '' or u.user_name ilike ?)\n     ", user, user)
    while True:
        rows = cursor.fetchmany(100)
        if not rows:
            break
        for row in rows:
            user_key = row.user_name.lower()
            facts[user_key] = {'name': row.user_name, 'locked': str(row.is_locked), 'password': row.password, 'expired': str(row.is_expired), 'profile': row.profile_name, 'resource_pool': row.resource_pool, 'roles': [], 'default_roles': []}
            if row.is_locked:
                facts[user_key]['locked_time'] = str(row.lock_time)
            if row.all_roles:
                facts[user_key]['roles'] = row.all_roles.replace(' ', '').split(',')
            if row.default_roles:
                facts[user_key]['default_roles'] = row.default_roles.replace(' ', '').split(',')
    return facts