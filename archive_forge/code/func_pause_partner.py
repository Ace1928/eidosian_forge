from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.hpe.nimble.plugins.module_utils.hpe_nimble import __version__ as NIMBLE_ANSIBLE_VERSION
import ansible_collections.hpe.nimble.plugins.module_utils.hpe_nimble as utils
def pause_partner(client_obj, downstream_hostname):
    if utils.is_null_or_empty(downstream_hostname):
        return (False, False, 'Pause replication partner failed as no downstream partner is provided.', {})
    try:
        upstream_repl_resp = client_obj.replication_partners.get(id=None, hostname=downstream_hostname)
        if utils.is_null_or_empty(upstream_repl_resp):
            return (False, False, f"Replication partner '{downstream_hostname}' cannot be paused as it is not present.", {})
        if upstream_repl_resp.attrs.get('paused') is False:
            client_obj.replication_partners.pause(id=upstream_repl_resp.attrs.get('id'))
            return (True, True, f"Paused replication partner '{downstream_hostname}' successfully.", {})
        else:
            return (True, False, f"Replication partner '{downstream_hostname}' is already in paused state.", {})
    except Exception as ex:
        return (False, False, f'Pause replication partner failed |{ex}', {})