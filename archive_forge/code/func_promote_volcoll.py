from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.hpe.nimble.plugins.module_utils.hpe_nimble import __version__ as NIMBLE_ANSIBLE_VERSION
import ansible_collections.hpe.nimble.plugins.module_utils.hpe_nimble as utils
def promote_volcoll(client_obj, volcoll_name):
    if utils.is_null_or_empty(volcoll_name):
        return (False, False, 'Promote volume collection failed as volume collection name is null.', {})
    try:
        volcoll_resp = client_obj.volume_collections.get(id=None, name=volcoll_name)
        if utils.is_null_or_empty(volcoll_resp):
            return (False, False, f"Volume collection '{volcoll_name}' not present to promote.", {})
        else:
            client_obj.volume_collections.promote(id=volcoll_resp.attrs.get('id'))
            return (True, True, f"Promoted volume collection '{volcoll_name}' successfully.", {})
    except Exception as ex:
        return (False, False, f'Promote volume collection failed | {ex}', {})