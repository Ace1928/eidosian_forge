from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.bgp_utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import to_request
from ansible.module_utils.connection import ConnectionError
from copy import deepcopy
def delete_specific_param_request(self, vrf_name, cmd):
    requests = []
    delete_static_path = '%s=%s/%s' % (self.network_instance_path, vrf_name, self.protocol_bgp_path)
    delete_static_path = delete_static_path + '/neighbors/neighbor=%s' % cmd['neighbor']
    if cmd.get('remote_as', None) is not None:
        if cmd['remote_as'].get('peer_as', None) is not None:
            delete_path = delete_static_path + '/config/peer-as'
            requests.append({'path': delete_path, 'method': DELETE})
        elif cmd['remote_as'].get('peer_type', None) is not None:
            delete_path = delete_static_path + '/config/peer-type'
            requests.append({'path': delete_path, 'method': DELETE})
    if cmd.get('peer_group', None) is not None:
        delete_path = delete_static_path + '/config/peer-group'
        requests.append({'path': delete_path, 'method': DELETE})
    if cmd.get('nbr_description', None) is not None:
        delete_path = delete_static_path + '/config/description'
        requests.append({'path': delete_path, 'method': DELETE})
    if cmd.get('disable_connected_check', None) is not None:
        delete_path = delete_static_path + '/config/disable-ebgp-connected-route-check'
        requests.append({'path': delete_path, 'method': DELETE})
    if cmd.get('dont_negotiate_capability', None) is not None:
        delete_path = delete_static_path + '/config/dont-negotiate-capability'
        requests.append({'path': delete_path, 'method': DELETE})
    if cmd.get('enforce_first_as', None) is not None:
        delete_path = delete_static_path + '/config/enforce-first-as'
        requests.append({'path': delete_path, 'method': DELETE})
    if cmd.get('enforce_multihop', None) is not None:
        delete_path = delete_static_path + '/config/enforce-multihop'
        requests.append({'path': delete_path, 'method': DELETE})
    if cmd.get('override_capability', None) is not None:
        delete_path = delete_static_path + '/config/override-capability'
        requests.append({'path': delete_path, 'method': DELETE})
    if cmd.get('port', None) is not None:
        delete_path = delete_static_path + '/config/peer-port'
        requests.append({'path': delete_path, 'method': DELETE})
    if cmd.get('shutdown_msg', None) is not None:
        delete_path = delete_static_path + '/config/shutdown-message'
        requests.append({'path': delete_path, 'method': DELETE})
    if cmd.get('solo', None) is not None:
        delete_path = delete_static_path + '/config/solo-peer'
        requests.append({'path': delete_path, 'method': DELETE})
    if cmd.get('strict_capability_match', None) is not None:
        delete_path = delete_static_path + '/config/strict-capability-match'
        requests.append({'path': delete_path, 'method': DELETE})
    if cmd.get('ttl_security', None) is not None:
        delete_path = delete_static_path + '/config/ttl-security-hops'
        requests.append({'path': delete_path, 'method': DELETE})
    if cmd.get('v6only', None) is not None:
        delete_path = delete_static_path + '/config/openconfig-bgp-ext:v6only'
        requests.append({'path': delete_path, 'method': DELETE})
    if cmd.get('local_as', None) is not None:
        if cmd['local_as'].get('as', None) is not None:
            delete_path = delete_static_path + '/config/local-as'
            requests.append({'path': delete_path, 'method': DELETE})
        if cmd['local_as'].get('no_prepend', None) is not None:
            delete_path = delete_static_path + '/config/local-as-no-prepend'
            requests.append({'path': delete_path, 'method': DELETE})
        if cmd['local_as'].get('replace_as', None) is not None:
            delete_path = delete_static_path + '/config/local-as-replace-as'
            requests.append({'path': delete_path, 'method': DELETE})
    if cmd.get('local_address', None) is not None:
        delete_path = delete_static_path + '/transport/config/local-address'
        requests.append({'path': delete_path, 'method': DELETE})
    if cmd.get('passive', None) is not None:
        delete_path = delete_static_path + '/transport/config/passive-mode'
        requests.append({'path': delete_path, 'method': DELETE})
    if cmd.get('advertisement_interval', None) is not None:
        delete_path = delete_static_path + '/timers/config/minimum-advertisement-interval'
        requests.append({'path': delete_path, 'method': DELETE})
    if cmd.get('timers', None) is not None:
        if cmd['timers'].get('holdtime', None) is not None:
            delete_path = delete_static_path + '/timers/config/hold-time'
            requests.append({'path': delete_path, 'method': DELETE})
        if cmd['timers'].get('keepalive', None) is not None:
            delete_path = delete_static_path + '/timers/config/keepalive-interval'
            requests.append({'path': delete_path, 'method': DELETE})
        if cmd['timers'].get('connect_retry', None) is not None:
            delete_path = delete_static_path + '/timers/config/connect-retry'
            requests.append({'path': delete_path, 'method': DELETE})
    if cmd.get('capability', None) is not None:
        if cmd['capability'].get('dynamic', None) is not None:
            delete_path = delete_static_path + '/config/capability-dynamic'
            requests.append({'path': delete_path, 'method': DELETE})
        if cmd['capability'].get('extended_nexthop', None) is not None:
            delete_path = delete_static_path + '/config/capability-extended-nexthop'
            requests.append({'path': delete_path, 'method': DELETE})
    if cmd.get('bfd', None) is not None:
        if cmd['bfd'].get('enabled', None) is not None:
            delete_path = delete_static_path + '/enable-bfd/config/enabled'
            requests.append({'path': delete_path, 'method': DELETE})
        if cmd['bfd'].get('check_failure', None) is not None:
            delete_path = delete_static_path + '/enable-bfd/config/check-control-plane-failure'
            requests.append({'path': delete_path, 'method': DELETE})
        if cmd['bfd'].get('profile', None) is not None:
            delete_path = delete_static_path + '/enable-bfd/config/bfd-profile'
            requests.append({'path': delete_path, 'method': DELETE})
    if cmd.get('auth_pwd', None) is not None:
        if cmd['auth_pwd'].get('pwd', None) is not None:
            delete_path = delete_static_path + '/auth-password/config/password'
            requests.append({'path': delete_path, 'method': DELETE})
        if cmd['auth_pwd'].get('encrypted', None) is not None:
            delete_path = delete_static_path + '/auth-password/config/encrypted'
            requests.append({'path': delete_path, 'method': DELETE})
    if cmd.get('ebgp_multihop', None) is not None:
        if cmd['ebgp_multihop'].get('enabled', None) is not None:
            delete_path = delete_static_path + '/ebgp-multihop/config/enabled'
            requests.append({'path': delete_path, 'method': DELETE})
        if cmd['ebgp_multihop'].get('multihop_ttl', None) is not None:
            delete_path = delete_static_path + '/ebgp-multihop/config/multihop-ttl'
            requests.append({'path': delete_path, 'method': DELETE})
    return requests