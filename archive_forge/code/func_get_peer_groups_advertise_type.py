from __future__ import (absolute_import, division, print_function)
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import exec_command, load_config
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import ce_argument_spec
def get_peer_groups_advertise_type(self):
    """get evpn peer group name advertise type list"""
    if len(self.config_list) != 2:
        return None
    self.config_list = self.config.split('l2vpn-family evpn')
    get1 = re.findall('group (\\S+) external', self.config_list[0])
    get2 = re.findall('group (\\S+) internal', self.config_list[0])
    if not get1 and (not get2):
        return None
    else:
        peer_groups = list()
        for item in get1:
            cmd = 'peer %s advertise arp' % item
            exist1 = is_config_exist(self.config_list[1], cmd)
            cmd = 'peer %s advertise irb' % item
            exist2 = is_config_exist(self.config_list[1], cmd)
            if exist1:
                peer_groups.append(dict(peer_group_name=item, advertise_type='arp'))
            if exist2:
                peer_groups.append(dict(peer_group_name=item, advertise_type='irb'))
        for item in get2:
            cmd = 'peer %s advertise arp' % item
            exist1 = is_config_exist(self.config_list[1], cmd)
            cmd = 'peer %s advertise irb' % item
            exist2 = is_config_exist(self.config_list[1], cmd)
            if exist1:
                peer_groups.append(dict(peer_group_name=item, advertise_type='arp'))
            if exist2:
                peer_groups.append(dict(peer_group_name=item, advertise_type='irb'))
        return peer_groups