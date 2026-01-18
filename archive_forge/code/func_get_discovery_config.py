from __future__ import (absolute_import, division, print_function)
import json
import time
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import strip_substr_dict
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
def get_discovery_config(module, rest_obj):
    disc_cfg_list = []
    proto_dev_map, dev_id_map = get_protocol_device_map(rest_obj)
    discovery_config_list = module.params.get('discovery_config_targets')
    for disc_config in discovery_config_list:
        disc_cfg = {}
        disc_cfg['DeviceType'] = list((dev_id_map[dev] for dev in disc_config.get('device_types') if dev in dev_id_map.keys()))
        devices = list(set(disc_config.get('device_types')))
        if len(devices) != len(disc_cfg['DeviceType']):
            invalid_dev = set(devices) - set(dev_id_map.keys())
            module.fail_json(msg=INVALID_DEVICES.format(','.join(invalid_dev)))
        disc_cfg['DiscoveryConfigTargets'] = list(({'NetworkAddressDetail': ip} for ip in disc_config['network_address_detail']))
        conn_profile = get_connection_profile(disc_config)
        given_protos = list((x['type'] for x in conn_profile['credentials']))
        req_protos = []
        for dev in disc_config.get('device_types'):
            proto_dev_value = proto_dev_map.get(dev, [])
            req_protos.extend(proto_dev_value)
        if not set(req_protos) & set(given_protos):
            module.fail_json(msg=ATLEAST_ONE_PROTOCOL, discovery_status=proto_dev_map)
        disc_cfg['ConnectionProfile'] = json.dumps(conn_profile)
        disc_cfg_list.append(disc_cfg)
    return disc_cfg_list