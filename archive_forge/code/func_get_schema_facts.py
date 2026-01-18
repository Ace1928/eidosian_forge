from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.text.converters import to_native
def get_schema_facts(cursor, schema=''):
    facts = {}
    cursor.execute("\n        select schema_name, schema_owner, create_time\n        from schemata\n        where not is_system_schema and schema_name not in ('public', 'TxtIndex')\n        and (? = '' or schema_name ilike ?)\n    ", schema, schema)
    while True:
        rows = cursor.fetchmany(100)
        if not rows:
            break
        for row in rows:
            facts[row.schema_name.lower()] = {'name': row.schema_name, 'owner': row.schema_owner, 'create_time': str(row.create_time), 'usage_roles': [], 'create_roles': []}
    cursor.execute("\n        select g.object_name as schema_name, r.name as role_name,\n        lower(g.privileges_description) privileges_description\n        from roles r join grants g\n        on g.grantee_id = r.role_id and g.object_type='SCHEMA'\n        and g.privileges_description like '%USAGE%'\n        and g.grantee not in ('public', 'dbadmin')\n        and (? = '' or g.object_name ilike ?)\n    ", schema, schema)
    while True:
        rows = cursor.fetchmany(100)
        if not rows:
            break
        for row in rows:
            schema_key = row.schema_name.lower()
            if 'create' in row.privileges_description:
                facts[schema_key]['create_roles'].append(row.role_name)
            else:
                facts[schema_key]['usage_roles'].append(row.role_name)
    return facts