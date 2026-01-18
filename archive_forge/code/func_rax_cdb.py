from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.rax import rax_argument_spec, rax_required_together, rax_to_dict, setup_rax_module
def rax_cdb(module, state, name, flavor, volume, cdb_type, cdb_version, wait, wait_timeout):
    if state == 'present':
        save_instance(module, name, flavor, volume, cdb_type, cdb_version, wait, wait_timeout)
    elif state == 'absent':
        delete_instance(module, name, wait, wait_timeout)