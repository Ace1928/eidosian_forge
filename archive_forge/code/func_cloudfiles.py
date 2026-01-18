from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.rax import rax_argument_spec, rax_required_together, setup_rax_module
def cloudfiles(module, container_, state, meta_, clear_meta, typ, ttl, public, private, web_index, web_error):
    """ Dispatch from here to work with metadata or file objects """
    cf = pyrax.cloudfiles
    if cf is None:
        module.fail_json(msg='Failed to instantiate client. This typically indicates an invalid region or an incorrectly capitalized region name.')
    if typ == 'container':
        container(cf, module, container_, state, meta_, clear_meta, ttl, public, private, web_index, web_error)
    else:
        meta(cf, module, container_, state, meta_, clear_meta)