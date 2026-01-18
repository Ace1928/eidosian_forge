from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.rax import rax_argument_spec, rax_required_together, setup_rax_module
def put_meta(module, cf, container, src, dest, meta, clear_meta):
    """ Set metadata on a container, single file, or comma-separated list.
    Passing a true value to clear_meta clears the metadata stored in Cloud
    Files before setting the new metadata to the value of "meta".
    """
    if src and dest:
        module.fail_json(msg='Error: ambiguous instructions; files to set meta have been specified on both src and dest args')
    objs = dest or src
    objs = map(str.strip, objs.split(','))
    c = _get_container(module, cf, container)
    try:
        results = [c.get_object(obj).set_metadata(meta, clear=clear_meta) for obj in objs]
    except Exception as e:
        module.fail_json(msg=e.message)
    EXIT_DICT['container'] = c.name
    EXIT_DICT['success'] = True
    if results:
        EXIT_DICT['changed'] = True
        EXIT_DICT['num_changed'] = True
    module.exit_json(**EXIT_DICT)