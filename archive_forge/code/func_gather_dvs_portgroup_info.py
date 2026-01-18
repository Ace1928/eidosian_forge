from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
from ansible.module_utils.six.moves.urllib.parse import unquote
def gather_dvs_portgroup_info(self):
    dvs_lists = self.dvsls
    result = dict()
    for dvs in dvs_lists:
        result[dvs.name] = list()
        for dvs_pg in dvs.portgroup:
            mac_learning = dict()
            network_policy = dict()
            teaming_policy = dict()
            port_policy = dict()
            vlan_info = dict()
            active_uplinks = list()
            standby_uplinks = list()
            if dvs_pg.config.type == 'ephemeral':
                port_binding = 'ephemeral'
            else:
                port_binding = 'static'
            if dvs_pg.config.autoExpand is True:
                port_allocation = 'elastic'
            else:
                port_allocation = 'fixed'
            if self.module.params['show_network_policy']:
                network_policy = dict(forged_transmits=dvs_pg.config.defaultPortConfig.macManagementPolicy.forgedTransmits, promiscuous=dvs_pg.config.defaultPortConfig.macManagementPolicy.allowPromiscuous, mac_changes=dvs_pg.config.defaultPortConfig.macManagementPolicy.macChanges)
            if self.module.params['show_mac_learning']:
                macLearningPolicy = dvs_pg.config.defaultPortConfig.macManagementPolicy.macLearningPolicy
                mac_learning = dict(allow_unicast_flooding=macLearningPolicy.allowUnicastFlooding, enabled=macLearningPolicy.enabled, limit=macLearningPolicy.limit, limit_policy=macLearningPolicy.limitPolicy)
            if self.module.params['show_teaming_policy']:
                teaming_policy = dict(policy=dvs_pg.config.defaultPortConfig.uplinkTeamingPolicy.policy.value, inbound_policy=dvs_pg.config.defaultPortConfig.uplinkTeamingPolicy.reversePolicy.value, notify_switches=dvs_pg.config.defaultPortConfig.uplinkTeamingPolicy.notifySwitches.value, rolling_order=dvs_pg.config.defaultPortConfig.uplinkTeamingPolicy.rollingOrder.value)
            if self.module.params['show_uplinks'] and dvs_pg.config.defaultPortConfig.uplinkTeamingPolicy and dvs_pg.config.defaultPortConfig.uplinkTeamingPolicy.uplinkPortOrder:
                active_uplinks = dvs_pg.config.defaultPortConfig.uplinkTeamingPolicy.uplinkPortOrder.activeUplinkPort
                standby_uplinks = dvs_pg.config.defaultPortConfig.uplinkTeamingPolicy.uplinkPortOrder.standbyUplinkPort
            if self.params['show_port_policy']:
                port_policy = dict(block_override=dvs_pg.config.policy.blockOverrideAllowed, ipfix_override=dvs_pg.config.policy.ipfixOverrideAllowed, live_port_move=dvs_pg.config.policy.livePortMovingAllowed, network_rp_override=dvs_pg.config.policy.networkResourcePoolOverrideAllowed, port_config_reset_at_disconnect=dvs_pg.config.policy.portConfigResetAtDisconnect, security_override=dvs_pg.config.policy.macManagementOverrideAllowed, shaping_override=dvs_pg.config.policy.shapingOverrideAllowed, traffic_filter_override=dvs_pg.config.policy.trafficFilterOverrideAllowed, uplink_teaming_override=dvs_pg.config.policy.uplinkTeamingOverrideAllowed, vendor_config_override=dvs_pg.config.policy.vendorConfigOverrideAllowed, vlan_override=dvs_pg.config.policy.vlanOverrideAllowed)
            if self.params['show_vlan_info']:
                vlan_info = self.get_vlan_info(dvs_pg.config.defaultPortConfig.vlan)
            dvpg_details = dict(portgroup_name=unquote(dvs_pg.name), moid=dvs_pg._moId, num_ports=dvs_pg.config.numPorts, dvswitch_name=dvs_pg.config.distributedVirtualSwitch.name, description=dvs_pg.config.description, type=dvs_pg.config.type, port_binding=port_binding, port_allocation=port_allocation, teaming_policy=teaming_policy, port_policy=port_policy, mac_learning=mac_learning, network_policy=network_policy, vlan_info=vlan_info, key=dvs_pg.key, active_uplinks=active_uplinks, standby_uplinks=standby_uplinks)
            result[dvs.name].append(dvpg_details)
    return result