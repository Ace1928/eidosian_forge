from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.rax import rax_argument_spec, rax_required_together, rax_to_dict, setup_rax_module
def save_database(module, cdb_id, name, character_set, collate):
    cdb = pyrax.cloud_databases
    try:
        instance = cdb.get(cdb_id)
    except Exception as e:
        module.fail_json(msg='%s' % e.message)
    changed = False
    database = find_database(instance, name)
    if not database:
        try:
            database = instance.create_database(name=name, character_set=character_set, collate=collate)
        except Exception as e:
            module.fail_json(msg='%s' % e.message)
        else:
            changed = True
    module.exit_json(changed=changed, action='create', database=rax_to_dict(database))