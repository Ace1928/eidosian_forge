from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, \
def config_connect_port_xml(self):
    """ Config connect port by xml """
    if 'connect port' in self.cur_cli_cfg.keys():
        if self.cur_cli_cfg['connect port'] == self.connect_port:
            pass
        else:
            cmd = 'snmp-agent udp-port %s' % self.connect_port
            cmds = list()
            cmds.append(cmd)
            self.updates_cmd.append(cmd)
            conf_str = CE_MERGE_SNMP_PORT % self.connect_port
            self.netconf_set_config(conf_str=conf_str)
            self.changed = True
    else:
        cmd = 'snmp-agent udp-port %s' % self.connect_port
        cmds = list()
        cmds.append(cmd)
        self.updates_cmd.append(cmd)
        conf_str = CE_MERGE_SNMP_PORT % self.connect_port
        self.netconf_set_config(conf_str=conf_str)
        self.changed = True