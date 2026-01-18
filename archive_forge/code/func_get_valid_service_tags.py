from __future__ import (absolute_import, division, print_function)
import json
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import \
def get_valid_service_tags(module, rest_obj):
    service_tags = []
    nic_configs = module.params.get('nic_configuration')
    if nic_configs:
        nic_ids = [nic.get('nic_identifier') for nic in nic_configs]
        if len(nic_ids) > len(set(nic_ids)):
            module.exit_json(failed=True, msg=DUPLICATE_NIC_IDENTIFIED)
    dev_map = get_rest_items(rest_obj, uri=DEVICE_URI)
    if module.params.get('device_service_tag'):
        cmp_set = set(module.params.get('device_service_tag')) - set(dict(dev_map).values())
        if cmp_set:
            module.exit_json(failed=True, msg=INVALID_DEV_ST.format(','.join(cmp_set)))
        service_tags = list(set(module.params.get('device_service_tag')))
    if module.params.get('device_id'):
        cmp_set = set(module.params.get('device_id')) - set(dict(dev_map).keys())
        if cmp_set:
            module.exit_json(failed=True, msg=INVALID_DEV_ID.format(','.join(map(str, cmp_set))))
        service_tags = [dev_map.get(id) for id in set(module.params.get('device_id'))]
    return service_tags