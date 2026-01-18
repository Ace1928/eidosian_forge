from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import ce_argument_spec, get_nc_config, set_nc_config, check_ip_addr
def delete_merge_syslog_channel(self, channel_id, channel_name):
    """delete channel id"""
    change_flag = False
    if channel_name:
        for id2name in self.channel_info['channelInfos']:
            channel_default_name = get_channel_name_default(id2name['icChnlId'])
            if id2name['icChnlId'] == channel_id and id2name['icChnlCfgName'] == channel_name:
                channel_name = channel_default_name
                change_flag = True
    if not channel_name:
        for id2name in self.channel_info['channelInfos']:
            channel_default_name = get_channel_name_default(id2name['icChnlId'])
            if id2name['icChnlId'] == channel_id and id2name['icChnlCfgName'] != channel_default_name:
                channel_name = channel_default_name
                change_flag = True
    if change_flag:
        conf_str = CE_NC_MERGE_CHANNEL_INFO_HEADER
        if channel_id:
            conf_str += '<icChnlId>%s</icChnlId>' % channel_id
        if channel_name:
            conf_str += '<icChnlCfgName>%s</icChnlCfgName>' % channel_name
        conf_str += CE_NC_MERGE_CHANNEL_INFO_TAIL
        recv_xml = set_nc_config(self.module, conf_str)
        if '<ok/>' not in recv_xml:
            self.module.fail_json(msg='Error: Merge syslog channel id failed.')
        self.updates_cmd.append('undo info-center channel %s' % channel_id)
        self.changed = True