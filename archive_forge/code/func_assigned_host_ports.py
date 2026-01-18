from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils._text import to_native
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule
def assigned_host_ports(self, apply_unassigning=False):
    """Determine if the hostPorts requested have already been assigned and return list of required used ports."""
    used_host_ports = {}
    for host in self.all_hosts:
        if host['label'].lower() != self.name.lower():
            for host_port in host['hostSidePorts']:
                for port in self.ports:
                    if port['port'] == host_port['address'] or port['label'].lower() == host_port['label'].lower():
                        if not self.force_port:
                            self.module.fail_json(msg='Port label or address is already used and force_port option is set to false!')
                        else:
                            port_ref = [port['hostPortRef'] for port in host['ports'] if port['hostPortName'] == host_port['address']]
                            port_ref.extend([port['initiatorRef'] for port in host['initiators'] if port['nodeName']['iscsiNodeName'] == host_port['address']])
                            if host['hostRef'] not in used_host_ports.keys():
                                used_host_ports.update({host['hostRef']: port_ref})
                            else:
                                used_host_ports[host['hostRef']].extend(port_ref)
    if apply_unassigning:
        for host_ref in used_host_ports.keys():
            try:
                rc, resp = self.request('storage-systems/%s/hosts/%s' % (self.ssid, host_ref), method='POST', data={'portsToRemove': used_host_ports[host_ref]})
            except Exception as err:
                self.module.fail_json(msg='Failed to unassign host port. Host Id [%s]. Array Id [%s]. Ports [%s]. Error [%s].' % (self.host_obj['id'], self.ssid, used_host_ports[host_ref], to_native(err)))