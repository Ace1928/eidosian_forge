from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.hpe.nimble.plugins.module_utils.hpe_nimble import __version__ as NIMBLE_ANSIBLE_VERSION
import ansible_collections.hpe.nimble.plugins.module_utils.hpe_nimble as utils
def delete_snapcoll(client_obj, snapcoll_name, volcoll_name):
    if utils.is_null_or_empty(snapcoll_name):
        return (False, False, 'Snapshot collection deletion failed as snapshot collection name is not present.', {})
    try:
        snapcoll_resp = client_obj.snapshot_collections.get(id=None, name=snapcoll_name, volcoll_name=volcoll_name)
        if utils.is_null_or_empty(snapcoll_resp):
            return (False, False, f"Snapshot collection '{snapcoll_name}' for volume collection '{volcoll_name}' not present to delete.", {})
        else:
            client_obj.snapshot_collections.delete(id=snapcoll_resp.attrs.get('id'))
            return (True, True, f"Snapshot collection '{snapcoll_name}' for volume collection '{volcoll_name}' deleted successfully.", {})
    except Exception as ex:
        return (False, False, f'Snapshot collection deletion failed | {ex}', {})