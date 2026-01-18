from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.text.converters import to_native
def get_node_facts(cursor, schema=''):
    facts = {}
    cursor.execute('\n        select node_name, node_address, export_address, node_state, node_type,\n            catalog_path\n        from nodes\n    ')
    while True:
        rows = cursor.fetchmany(100)
        if not rows:
            break
        for row in rows:
            facts[row.node_address] = {'node_name': row.node_name, 'export_address': row.export_address, 'node_state': row.node_state, 'node_type': row.node_type, 'catalog_path': row.catalog_path}
    return facts