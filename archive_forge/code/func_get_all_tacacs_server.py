from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.argspec.tacacs_server.tacacs_server import Tacacs_serverArgs
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible.module_utils.connection import ConnectionError
def get_all_tacacs_server(self):
    """Get all the tacacs_server configured in the device"""
    request = [{'path': 'data/openconfig-system:system/aaa/server-groups/server-group=TACACS/config', 'method': GET}]
    tacacs_server_data = {}
    try:
        response = edit_config(self._module, to_request(self._module, request))
    except ConnectionError as exc:
        self._module.fail_json(msg=str(exc), code=exc.code)
    if 'openconfig-system:config' in response[0][1]:
        raw_tacacs_global_data = response[0][1].get('openconfig-system:config', {})
        if 'auth-type' in raw_tacacs_global_data:
            tacacs_server_data['auth_type'] = raw_tacacs_global_data['auth-type']
        if 'secret-key' in raw_tacacs_global_data:
            tacacs_server_data['key'] = raw_tacacs_global_data['secret-key']
        if 'source-interface' in raw_tacacs_global_data:
            tacacs_server_data['source_interface'] = raw_tacacs_global_data['source-interface']
        if 'timeout' in raw_tacacs_global_data:
            tacacs_server_data['timeout'] = raw_tacacs_global_data['timeout']
    request = [{'path': 'data/openconfig-system:system/aaa/server-groups/server-group=TACACS/servers', 'method': GET}]
    hosts = []
    try:
        response = edit_config(self._module, to_request(self._module, request))
    except ConnectionError as exc:
        self._module.fail_json(msg=str(exc), code=exc.code)
    raw_tacacs_server_list = []
    if 'openconfig-system:servers' in response[0][1]:
        raw_tacacs_server_list = response[0][1].get('openconfig-system:servers', {}).get('server', [])
    for tacacs_host in raw_tacacs_server_list:
        host_data = {}
        if 'address' in tacacs_host:
            host_data['name'] = tacacs_host['address']
            cfg = tacacs_host.get('config', None)
            if cfg:
                if 'auth-type' in cfg:
                    host_data['auth_type'] = cfg['auth-type']
                if 'priority' in cfg:
                    host_data['priority'] = cfg['priority']
                if 'vrf' in cfg:
                    host_data['vrf'] = cfg['vrf']
                if 'timeout' in cfg:
                    host_data['timeout'] = cfg['timeout']
            if tacacs_host.get('tacacs', None) and tacacs_host['tacacs'].get('config', None):
                tacas_cfg = tacacs_host['tacacs']['config']
                if tacas_cfg.get('port', None):
                    host_data['port'] = tacas_cfg['port']
                if tacas_cfg.get('secret-key', None):
                    host_data['key'] = tacas_cfg['secret-key']
        if host_data:
            hosts.append(host_data)
    if hosts:
        tacacs_server_data['servers'] = {'host': hosts}
    return tacacs_server_data