from __future__ import absolute_import, division, print_function
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible.module_utils.connection import ConnectionError
def get_peergroups(module, vrf_name):
    peer_groups = []
    request_path = '%s=%s/protocols/protocol=BGP,bgp/bgp/peer-groups' % (network_instance_path, vrf_name)
    request = {'path': request_path, 'method': GET}
    try:
        response = edit_config(module, to_request(module, request))
    except ConnectionError as exc:
        module.fail_json(msg=str(exc), code=exc.code)
    resp = response[0][1]
    if 'openconfig-network-instance:peer-groups' in resp:
        data = resp['openconfig-network-instance:peer-groups']
        if 'peer-group' in data:
            for peer_group in data['peer-group']:
                pg = {}
                if 'config' in peer_group:
                    if 'peer-group-name' in peer_group['config']:
                        pg.update({'name': peer_group['config']['peer-group-name']})
                    if 'description' in peer_group['config']:
                        pg.update({'pg_description': peer_group['config']['description']})
                    if 'disable-ebgp-connected-route-check' in peer_group['config']:
                        pg.update({'disable_connected_check': peer_group['config']['disable-ebgp-connected-route-check']})
                    if 'dont-negotiate-capability' in peer_group['config']:
                        pg.update({'dont_negotiate_capability': peer_group['config']['dont-negotiate-capability']})
                    if 'enforce-first-as' in peer_group['config']:
                        pg.update({'enforce_first_as': peer_group['config']['enforce-first-as']})
                    if 'enforce-multihop' in peer_group['config']:
                        pg.update({'enforce_multihop': peer_group['config']['enforce-multihop']})
                    local_as = {}
                    if 'local-as' in peer_group['config']:
                        local_as.update({'as': peer_group['config']['local-as']})
                    if 'local-as-no-prepend' in peer_group['config']:
                        local_as.update({'no_prepend': peer_group['config']['local-as-no-prepend']})
                    if 'local-as-replace-as' in peer_group['config']:
                        local_as.update({'replace_as': peer_group['config']['local-as-replace-as']})
                    if 'override-capability' in peer_group['config']:
                        pg.update({'override_capability': peer_group['config']['override-capability']})
                    if 'shutdown-message' in peer_group['config']:
                        pg.update({'shutdown_msg': peer_group['config']['shutdown-message']})
                    if 'solo-peer' in peer_group['config']:
                        pg.update({'solo': peer_group['config']['solo-peer']})
                    if 'strict-capability-match' in peer_group['config']:
                        pg.update({'strict_capability_match': peer_group['config']['strict-capability-match']})
                    if 'ttl-security-hops' in peer_group['config']:
                        pg.update({'ttl_security': peer_group['config']['ttl-security-hops']})
                auth_pwd = {}
                if 'auth-password' in peer_group and 'config' in peer_group['auth-password']:
                    if 'encrypted' in peer_group['auth-password']['config']:
                        auth_pwd.update({'encrypted': peer_group['auth-password']['config']['encrypted']})
                    if 'password' in peer_group['auth-password']['config']:
                        auth_pwd.update({'pwd': peer_group['auth-password']['config']['password']})
                bfd = {}
                if 'enable-bfd' in peer_group and 'config' in peer_group['enable-bfd']:
                    if 'enabled' in peer_group['enable-bfd']['config']:
                        bfd.update({'enabled': peer_group['enable-bfd']['config']['enabled']})
                    if 'check-control-plane-failure' in peer_group['enable-bfd']['config']:
                        bfd.update({'check_failure': peer_group['enable-bfd']['config']['check-control-plane-failure']})
                    if 'bfd-profile' in peer_group['enable-bfd']['config']:
                        bfd.update({'profile': peer_group['enable-bfd']['config']['bfd-profile']})
                ebgp_multihop = {}
                if 'ebgp-multihop' in peer_group and 'config' in peer_group['ebgp-multihop']:
                    if 'enabled' in peer_group['ebgp-multihop']['config']:
                        ebgp_multihop.update({'enabled': peer_group['ebgp-multihop']['config']['enabled']})
                    if 'multihop-ttl' in peer_group['ebgp-multihop']['config']:
                        ebgp_multihop.update({'multihop_ttl': peer_group['ebgp-multihop']['config']['multihop-ttl']})
                if 'transport' in peer_group and 'config' in peer_group['transport']:
                    if 'local-address' in peer_group['transport']['config']:
                        pg.update({'local_address': peer_group['transport']['config']['local-address']})
                    if 'passive-mode' in peer_group['transport']['config']:
                        pg.update({'passive': peer_group['transport']['config']['passive-mode']})
                if 'timers' in peer_group and 'config' in peer_group['timers']:
                    if 'minimum-advertisement-interval' in peer_group['timers']['config']:
                        pg.update({'advertisement_interval': peer_group['timers']['config']['minimum-advertisement-interval']})
                timers = {}
                if 'hold-time' in peer_group['timers']['config']:
                    timers.update({'holdtime': peer_group['timers']['config']['hold-time']})
                if 'keepalive-interval' in peer_group['timers']['config']:
                    timers.update({'keepalive': peer_group['timers']['config']['keepalive-interval']})
                if 'connect-retry' in peer_group['timers']['config']:
                    timers.update({'connect_retry': peer_group['timers']['config']['connect-retry']})
                capability = {}
                if 'config' in peer_group and 'capability-dynamic' in peer_group['config']:
                    capability.update({'dynamic': peer_group['config']['capability-dynamic']})
                if 'config' in peer_group and 'capability-extended-nexthop' in peer_group['config']:
                    capability.update({'extended_nexthop': peer_group['config']['capability-extended-nexthop']})
                remote_as = {}
                if 'config' in peer_group and 'peer-as' in peer_group['config']:
                    remote_as.update({'peer_as': peer_group['config']['peer-as']})
                if 'config' in peer_group and 'peer-type' in peer_group['config']:
                    remote_as.update({'peer_type': peer_group['config']['peer-type'].lower()})
                afis = []
                if 'afi-safis' in peer_group and 'afi-safi' in peer_group['afi-safis']:
                    for each in peer_group['afi-safis']['afi-safi']:
                        samp = {}
                        if 'afi-safi-name' in each and each['afi-safi-name']:
                            tmp = each['afi-safi-name'].split(':')
                            if tmp:
                                split_tmp = tmp[1].split('_')
                                if split_tmp:
                                    afi = split_tmp[0].lower()
                                    safi = split_tmp[1].lower()
                                if afi and safi:
                                    samp.update({'afi': afi})
                                    samp.update({'safi': safi})
                        if 'config' in each and 'enabled' in each['config']:
                            samp.update({'activate': each['config']['enabled']})
                        if 'allow-own-as' in each and 'config' in each['allow-own-as']:
                            allowas_in = {}
                            allowas_conf = each['allow-own-as']['config']
                            if 'origin' in allowas_conf and allowas_conf['origin']:
                                allowas_in.update({'origin': allowas_conf['origin']})
                            elif 'as-count' in allowas_conf and allowas_conf['as-count']:
                                allowas_in.update({'value': allowas_conf['as-count']})
                            if allowas_in:
                                samp.update({'allowas_in': allowas_in})
                        if 'ipv4-unicast' in each:
                            if 'config' in each['ipv4-unicast']:
                                ip_afi_conf = each['ipv4-unicast']['config']
                                ip_afi = update_bgp_nbr_pg_ip_afi_dict(ip_afi_conf)
                                if ip_afi:
                                    samp.update({'ip_afi': ip_afi})
                            if 'prefix-limit' in each['ipv4-unicast'] and 'config' in each['ipv4-unicast']['prefix-limit']:
                                pfx_lmt_conf = each['ipv4-unicast']['prefix-limit']['config']
                                prefix_limit = update_bgp_nbr_pg_prefix_limit_dict(pfx_lmt_conf)
                                if prefix_limit:
                                    samp.update({'prefix_limit': prefix_limit})
                        elif 'ipv6-unicast' in each:
                            if 'config' in each['ipv6-unicast']:
                                ip_afi_conf = each['ipv6-unicast']['config']
                                ip_afi = update_bgp_nbr_pg_ip_afi_dict(ip_afi_conf)
                                if ip_afi:
                                    samp.update({'ip_afi': ip_afi})
                            if 'prefix-limit' in each['ipv6-unicast'] and 'config' in each['ipv6-unicast']['prefix-limit']:
                                pfx_lmt_conf = each['ipv6-unicast']['prefix-limit']['config']
                                prefix_limit = update_bgp_nbr_pg_prefix_limit_dict(pfx_lmt_conf)
                                if prefix_limit:
                                    samp.update({'prefix_limit': prefix_limit})
                        if 'prefix-list' in each and 'config' in each['prefix-list']:
                            pfx_lst_conf = each['prefix-list']['config']
                            if 'import-policy' in pfx_lst_conf and pfx_lst_conf['import-policy']:
                                samp.update({'prefix_list_in': pfx_lst_conf['import-policy']})
                            if 'export-policy' in pfx_lst_conf and pfx_lst_conf['export-policy']:
                                samp.update({'prefix_list_out': pfx_lst_conf['export-policy']})
                        if samp:
                            afis.append(samp)
                if auth_pwd:
                    pg.update({'auth_pwd': auth_pwd})
                if bfd:
                    pg.update({'bfd': bfd})
                if ebgp_multihop:
                    pg.update({'ebgp_multihop': ebgp_multihop})
                if local_as:
                    pg.update({'local_as': local_as})
                if timers:
                    pg.update({'timers': timers})
                if capability:
                    pg.update({'capability': capability})
                if remote_as:
                    pg.update({'remote_as': remote_as})
                if afis and len(afis) > 0:
                    afis_dict = {}
                    afis_dict.update({'afis': afis})
                    pg.update({'address_family': afis_dict})
                peer_groups.append(pg)
    return peer_groups