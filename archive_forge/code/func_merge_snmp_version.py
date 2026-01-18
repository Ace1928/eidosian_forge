from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, \
def merge_snmp_version(self):
    """ Merge snmp version operation """
    conf_str = CE_MERGE_SNMP_VERSION % self.version
    recv_xml = self.netconf_set_config(conf_str=conf_str)
    if '<ok/>' not in recv_xml:
        self.module.fail_json(msg='Error: Merge snmp version failed.')
    if self.version == 'none':
        cmd = 'snmp-agent sys-info version %s disable' % self.cur_cli_cfg['version']
        self.updates_cmd.append(cmd)
    elif self.version == 'v1v2c':
        cmd = 'snmp-agent sys-info version v1'
        self.updates_cmd.append(cmd)
        cmd = 'snmp-agent sys-info version v2c'
        self.updates_cmd.append(cmd)
    elif self.version == 'v1v3':
        cmd = 'snmp-agent sys-info version v1'
        self.updates_cmd.append(cmd)
        cmd = 'snmp-agent sys-info version v3'
        self.updates_cmd.append(cmd)
    elif self.version == 'v2cv3':
        cmd = 'snmp-agent sys-info version v2c'
        self.updates_cmd.append(cmd)
        cmd = 'snmp-agent sys-info version v3'
        self.updates_cmd.append(cmd)
    else:
        cmd = 'snmp-agent sys-info version %s' % self.version
        self.updates_cmd.append(cmd)
    self.changed = True