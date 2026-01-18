from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.text.converters import to_native
def get_configuration_facts(cursor, parameter_name=''):
    facts = {}
    cursor.execute("\n        select c.parameter_name, c.current_value, c.default_value\n        from configuration_parameters c\n        where c.node_name = 'ALL'\n        and (? = '' or c.parameter_name ilike ?)\n    ", parameter_name, parameter_name)
    while True:
        rows = cursor.fetchmany(100)
        if not rows:
            break
        for row in rows:
            facts[row.parameter_name.lower()] = {'parameter_name': row.parameter_name, 'current_value': row.current_value, 'default_value': row.default_value}
    return facts