from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import ce_argument_spec, get_nc_config, set_nc_config, check_ip_addr
def delete_merge_out_direct(self, out_direct, channel_id):
    """delete out direct"""
    change_flag = False
    channel_id_default = get_out_direct_default(out_direct)
    if channel_id:
        for id2name in self.channel_direct_info['channelDirectInfos']:
            if id2name['icOutDirect'] == out_direct and id2name['icCfgChnlId'] == channel_id:
                if channel_id != channel_id_default:
                    channel_id = channel_id_default
                    change_flag = True
    if not channel_id:
        for id2name in self.channel_direct_info['channelDirectInfos']:
            if id2name['icOutDirect'] == out_direct and id2name['icCfgChnlId'] != channel_id_default:
                channel_id = channel_id_default
                change_flag = True
    if change_flag:
        conf_str = CE_NC_MERGE_CHANNEL_DIRECT_HEADER
        if out_direct:
            conf_str += '<icOutDirect>%s</icOutDirect>' % out_direct
        if channel_id:
            conf_str += '<icCfgChnlId>%s</icCfgChnlId>' % channel_id
        conf_str += CE_NC_MERGE_CHANNEL_DIRECT_TAIL
        recv_xml = set_nc_config(self.module, conf_str)
        if '<ok/>' not in recv_xml:
            self.module.fail_json(msg='Error: Merge syslog channel out direct failed.')
        self.updates_cmd.append('undo info-center logfile channel')
        self.changed = True