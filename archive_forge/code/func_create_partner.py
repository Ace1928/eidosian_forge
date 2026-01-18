from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.hpe.nimble.plugins.module_utils.hpe_nimble import __version__ as NIMBLE_ANSIBLE_VERSION
import ansible_collections.hpe.nimble.plugins.module_utils.hpe_nimble as utils
def create_partner(client_obj, downstream_hostname, **kwargs):
    if utils.is_null_or_empty(downstream_hostname):
        return (False, False, 'Create replication partner failed as name is not present.', {})
    try:
        upstream_repl_resp = client_obj.replication_partners.get(id=None, hostname=downstream_hostname)
        if utils.is_null_or_empty(upstream_repl_resp):
            params = utils.remove_null_args(**kwargs)
            upstream_repl_resp = client_obj.replication_partners.create(hostname=downstream_hostname, **params)
            return (True, True, f"Replication partner '{downstream_hostname}' created successfully.", {}, upstream_repl_resp.attrs)
        else:
            return (False, False, f"Replication partner '{downstream_hostname}' cannot be created as it is already present in given state.", {}, upstream_repl_resp.attrs)
    except Exception as ex:
        return (False, False, f'Replication partner creation failed |{ex}', {}, {})