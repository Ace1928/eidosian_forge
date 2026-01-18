from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.hpe.nimble.plugins.module_utils.hpe_nimble import __version__ as NIMBLE_ANSIBLE_VERSION
import ansible_collections.hpe.nimble.plugins.module_utils.hpe_nimble as utils
import re
def fetch_snapshots_for_all_subset(subset, client_obj):
    if subset is None or client_obj is None:
        return {}
    result = {}
    total_snap = []
    vol_list_resp = client_obj.volumes.list(detail=False)
    if vol_list_resp is not None and vol_list_resp.__len__() > 0:
        for vol_item in vol_list_resp:
            vol_name = vol_item.attrs.get('name')
            snap_list = client_obj.snapshots.list(detail=subset['detail'], vol_name=vol_name, limit=subset['limit'])
            if snap_list is not None and snap_list.__len__() > 0:
                total_snap.extend(snap_list)
                if subset['limit'] is not None and total_snap.__len__() >= subset['limit']:
                    total_snap = total_snap[0:subset['limit']]
                    break
        if total_snap.__len__() > 0:
            result['snapshots'] = generate_dict('snapshots', total_snap)['snapshots']
    return result