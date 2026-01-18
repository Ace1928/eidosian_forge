from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.netconf import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.junipernetworks.junos.plugins.module_utils.network.junos.facts.facts import (
from ansible_collections.junipernetworks.junos.plugins.module_utils.network.junos.junos import (
def render_af(self, w_af_list, family_root):
    if w_af_list:
        for waf in w_af_list:
            nlri_node = build_child_xml_node(family_root, waf['afi'])
            nlri_types = waf.get('af_type')
            for type in nlri_types:
                type_node = build_child_xml_node(nlri_node, type['type'])
                if 'accepted_prefix_limit' in type.keys():
                    apl = type.get('accepted_prefix_limit')
                    apl_node = build_child_xml_node(type_node, 'accepted-prefix-limit')
                    if 'maximum' in apl.keys():
                        build_child_xml_node(apl_node, 'maximum', apl['maximum'])
                    td_node = None
                    if 'limit_threshold' in apl.keys():
                        td_node = build_child_xml_node(apl_node, 'teardown')
                        build_child_xml_node(td_node, 'limit-threshold', apl.get('limit_threshold'))
                    elif 'teardown' in apl.keys():
                        td_node = build_child_xml_node(apl_node, 'teardown')
                    it_node = None
                    if 'idle_timeout_value' in apl.keys():
                        it_node = build_child_xml_node(td_node, 'idle-timeout')
                        build_child_xml_node(it_node, 'timeout', apl.get('idle_timeout_value'))
                    elif 'forever' in apl.keys():
                        if it_node is None:
                            it_node = build_child_xml_node(td_node, 'idle-timeout')
                        if it_node is not None:
                            it_node = build_child_xml_node(td_node, 'idle-timeout')
                        build_child_xml_node(it_node, 'forever')
                if 'add_path' in type.keys():
                    ap = type.get('add_path')
                    ap_node = build_child_xml_node(type_node, 'add-path')
                    if 'receive' in ap.keys():
                        build_child_xml_node(ap_node, 'receive')
                    if 'send' in ap.keys():
                        send = ap.get('send')
                        send_node = build_child_xml_node(ap_node, 'send')
                        if 'path_count' in send.keys():
                            build_child_xml_node(send_node, 'path-count', send.get('path_count'))
                        if 'include_backup_path' in send.keys():
                            build_child_xml_node(send_node, 'include-backup-path', send.get('include_backup_path'))
                        if 'path_selection_mode' in send.keys():
                            psm = send.get('path_selection_mode')
                            psm_node = build_child_xml_node(send_node, 'path-selection-mode')
                            if 'all_paths' in psm.keys():
                                build_child_xml_node(psm_node, 'all-paths')
                            if 'equal_cost_paths' in psm.keys():
                                build_child_xml_node(psm_node, 'equal-cost-paths')
                        if 'prefix_policy' in send.keys():
                            build_child_xml_node(send_node, 'prefix-policy', send.get('prefix_policy'))
                if 'aggregate_label' in type.keys():
                    al = type.get('aggregate_label')
                    al_node = build_child_xml_node(type_node, 'aggregate_label')
                    if 'community' in al.keys():
                        build_child_xml_node(al_node, 'community', al.get('community'))
                if 'aigp' in type.keys():
                    aigp = type.get('aigp')
                    if 'disable' in aigp.keys():
                        aigp_node = build_child_xml_node(type_node, 'aigp')
                        build_child_xml_node(aigp_node, 'disable')
                    else:
                        build_child_xml_node(type_node, 'aigp')
                if 'damping' in type.keys():
                    build_child_xml_node(type_node, 'damping')
                if 'defer_initial_multipath_build' in type.keys():
                    dimb = type.get('defer_initial_multipath_build')
                    dimb_node = build_child_xml_node(type_node, 'defer-initial-multipath-build')
                    if dimb and 'maximum_delay' in dimb.keys():
                        build_child_xml_node(dimb_node, 'maximum-delay', dimb.get('maximum_delay'))
                if 'delay_route_advertisements' in type.keys():
                    dra = type.get('delay_route_advertisements')
                    dra_node = build_child_xml_node(type_node, 'delay-route-advertisements')
                    if 'max_delay_route_age' in dra.keys() or 'max_delay_routing_uptime' in dra.keys():
                        maxd_node = build_child_xml_node(dra_node, 'maximum-delay')
                        if 'max_delay_route_age' in dra.keys():
                            build_child_xml_node(maxd_node, 'route-age', dra.get('max_delay_route_age'))
                        if 'max_delay_routing_uptime' in dra.keys():
                            build_child_xml_node(maxd_node, 'routing-uptime', dra.get('max_delay_routing_uptime'))
                    if 'min_delay_inbound_convergence' in dra.keys() or 'min_delay_routing_uptime' in dra.keys():
                        mind_node = build_child_xml_node(dra_node, 'minimum-delay')
                        if 'min_delay_inbound_convergence' in dra.keys():
                            build_child_xml_node(mind_node, 'inbound-convergence', dra.get('min_delay_inbound_convergence'))
                        if 'min_delay_routing_uptime' in dra.keys():
                            build_child_xml_node(mind_node, 'routing-uptime', dra.get('min_delay_routing_uptime'))
                if 'entropy_label' in type.keys():
                    el = type.get('entropy_label')
                    el_node = build_child_xml_node(type_node, 'entropy-label')
                    if 'import' in el.keys():
                        build_child_xml_node(el_node, 'import', el.get('import'))
                    if 'no_next_hop_validation' in el.keys():
                        build_child_xml_node(el_node, 'no-next-hop-validation')
                if 'explicit_null' in type.keys():
                    en = type.get('explicit_null')
                    if 'connected_only' in en.keys():
                        en_node = build_child_xml_node(type_node, 'explicit-null')
                        build_child_xml_node(en_node, 'connected-only')
                    else:
                        build_child_xml_node(type_node, 'explicit-null')
                if 'extended_nexthop' in type.keys():
                    enh = type.get('extended_nexthop')
                    if enh:
                        build_child_xml_node(type_node, 'extended-nexthop')
                if 'extended_nexthop_color' in type.keys():
                    enhc = type.get('extended_nexthop_color')
                    if enhc:
                        build_child_xml_node(type_node, 'extended-nexthop-color')
                if 'graceful_restart_forwarding_state_bit' in type.keys():
                    grfs = type.get('graceful_restart_forwarding_state_bit')
                    gr_node = build_child_xml_node(type_node, 'graceful-restart')
                    build_child_xml_node(gr_node, 'forwarding-state-bit', grfs)
                if 'local_ipv4_address' in type.keys():
                    build_child_xml_node(type_node, 'local-ipv4-address', type.get('local_ipv4_address'))
                if 'legacy_redirect_ip_action' in type.keys():
                    lria = type.get('legacy_redirect_ip_action')
                    lria_node = build_child_xml_node(type_node, 'legacy-redirect-ip-action')
                    if 'send' in lria.keys():
                        build_child_xml_node(lria_node, 'send')
                    if 'receive' in lria.keys():
                        build_child_xml_node(lria_node, 'receive')
                if 'loops' in type.keys():
                    build_child_xml_node(type_node, 'loops', type.get('loops'))
                if 'no_install' in type.keys():
                    if type.get('no_install'):
                        build_child_xml_node(type_node, 'no-install')
                if 'no_validate' in type.keys():
                    build_child_xml_node(type_node, 'no-validate', type.get('no_validate'))
                if 'output_queue_priority_expedited' in type.keys() or 'output_queue_priority_priority' in type.keys():
                    oqp_node = build_child_xml_node(type_node, 'output-queue-priority')
                    if 'output_queue_priority_expedited' in type.keys() and type.get('output_queue_priority_expedited'):
                        build_child_xml_node(oqp_node, 'expedited')
                    if 'output_queue_priority_priority' in type.keys():
                        build_child_xml_node(oqp_node, 'priority', type.get('output_queue_priority_priority'))
                if 'per_prefix_label' in type.keys():
                    if type.get('per_prefix_label'):
                        build_child_xml_node(type_node, 'per-prefix-label')
                if 'per_group_label' in type.keys():
                    if type.get('per_group_label'):
                        build_child_xml_node(type_node, 'per-group-label')
                if 'prefix_limit' in type.keys():
                    pl = type.get('prefix_limit')
                    pl_node = build_child_xml_node(type_node, 'prefix-limit')
                    if 'maximum' in pl.keys():
                        build_child_xml_node(pl_node, 'maximum', pl['maximum'])
                    td_node = None
                    if 'limit_threshold' in pl.keys():
                        td_node = build_child_xml_node(pl_node, 'teardown', pl.get('limit_threshold'))
                    elif 'teardown' in pl.keys():
                        td_node = build_child_xml_node(pl_node, 'teardown')
                    it_node = None
                    if 'idle_timeout_value' in pl.keys():
                        it_node = build_child_xml_node(td_node, 'idle-timeout', pl.get('idle_timeout_value'))
                    elif 'idle_timeout' in pl.keys():
                        it_node = build_child_xml_node(td_node, 'idle-timeout')
                    if 'forever' in pl.keys():
                        if it_node is None:
                            it_node = build_child_xml_node(td_node, 'idle-timeout')
                        build_child_xml_node(it_node, 'forever')
                if 'resolve_vpn' in type.keys():
                    if type.get('resolve_vpn'):
                        build_child_xml_node(type_node, 'resolve-vpn')
                if 'rib' in type.keys():
                    rib_node = build_child_xml_node(type_node, 'rib')
                    build_child_xml_node(rib_node, 'inet.3')
                if 'ribgroup_name' in type.keys():
                    build_child_xml_node(type_node, 'rib-group', type.get('ribgroup_name'))
                if 'route_refresh_priority_expedited' in type.keys() or 'route_refresh_priority_priority' in type.keys():
                    rrp_node = build_child_xml_node(type_node, 'route-refresh-priority')
                    if 'route_refresh_priority_expedited' in type.keys() and type.get('route_refresh_priority_expedited'):
                        build_child_xml_node(rrp_node, 'expedited')
                    if 'route_refresh_priority_priority' in type.keys():
                        build_child_xml_node(rrp_node, 'priority', type.get('route_refresh_priority_priority'))
                if 'secondary_independent_resolution' in type.keys():
                    if type.get('secondary_independent_resolution'):
                        build_child_xml_node(type_node, 'secondary-independent-resolution')
                if 'withdraw_priority_expedited' in type.keys() or 'withdraw_priority_priority' in type.keys():
                    wp_node = build_child_xml_node(type_node, 'withdraw-priority')
                    if 'withdraw_priority_expedited' in type.keys() and type.get('withdraw_priority_expedited'):
                        build_child_xml_node(wp_node, 'expedited')
                    if 'withdraw_priority_priority' in type.keys():
                        build_child_xml_node(wp_node, 'priority', type.get('withdraw_priority_priority'))
                if 'strip_nexthop' in type.keys():
                    if type.get('strip_nexthop'):
                        build_child_xml_node(type_node, 'strip-nexthop')
                if 'topology' in type.keys():
                    topologies = type.get('topology')
                    top_node = build_child_xml_node(type_node, 'topology')
                    for topology in topologies:
                        if 'name' in topology.keys():
                            build_child_xml_node(top_node, 'name', topology.get('name'))
                        if 'community' in topology.keys():
                            communities = topology.get('community')
                            for community in communities:
                                build_child_xml_node(top_node, 'community', community)
                if 'traffic_statistics' in type.keys():
                    ts = type.get('traffic_statistics')
                    ts_node = build_child_xml_node(type_node, 'traffic-statistics')
                    if 'interval' in ts.keys():
                        build_child_xml_node(ts_node, 'interval', ts.get('interval'))
                    if 'labeled_path' in ts.keys and ts.get('labeled_path'):
                        build_child_xml_node(ts_node, 'labeled-path')