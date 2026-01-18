from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
from ansible.module_utils._text import to_native
def set_desired_state(self):
    lag_uplinks = []
    switch_uplink_ports = {'non_lag': []}
    for dvs_host_member in self.dv_switch.config.host:
        if dvs_host_member.config.host.name == self.esxi_hostname:
            break
    portCriteria = vim.dvs.PortCriteria()
    portCriteria.host = [self.host]
    portCriteria.portgroupKey = self.uplink_portgroup.key
    portCriteria.uplinkPort = True
    ports = self.dv_switch.FetchDVPorts(portCriteria)
    for name, lag in self.lags.items():
        switch_uplink_ports[name] = []
        for uplinkName in lag.uplinkName:
            for port in ports:
                if port.config.name == uplinkName:
                    switch_uplink_ports[name].append(port.key)
                    lag_uplinks.append(port.key)
    for port in sorted(ports, key=lambda port: port.config.name):
        if port.key in self.uplink_portgroup.portKeys and port.key not in lag_uplinks:
            switch_uplink_ports['non_lag'].append(port.key)
    if self.vmnics is not None:
        count = 0
        for vmnic in self.vmnics:
            self.desired_state[vmnic] = switch_uplink_ports['non_lag'][count]
            count += 1
    else:
        for pnicSpec in dvs_host_member.config.backing.pnicSpec:
            if pnicSpec.uplinkPortKey not in lag_uplinks:
                self.desired_state[pnicSpec.pnicDevice] = pnicSpec.uplinkPortKey
    if self.lag_uplinks is not None:
        for lag in self.lag_uplinks:
            count = 0
            for vmnic in lag['vmnics']:
                self.desired_state[vmnic] = switch_uplink_ports[lag['lag']][count]
                count += 1
    else:
        for pnicSpec in dvs_host_member.config.backing.pnicSpec:
            if pnicSpec.uplinkPortKey in lag_uplinks:
                self.desired_state[pnicSpec.pnicDevice] = pnicSpec.uplinkPortKey