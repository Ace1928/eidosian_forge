from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def config_log_soruce(self):
    """config info-center sources"""
    xml_str = ''
    if not self.module_name or not self.channel_id:
        return xml_str
    source = self.log_dict['source']
    if self.state == 'present':
        xml_str = '<icSources><icSource operation="merge">'
        cmd = 'info-center source %s channel %s log' % (self.module_name, self.channel_id)
    else:
        if not source or self.module_name != source.get('moduleName').lower() or self.channel_id != source.get('icChannelId'):
            return ''
        if self.log_enable == 'no_use' and (not self.log_level):
            xml_str = '<icSources><icSource operation="delete">'
        else:
            xml_str = '<icSources><icSource operation="merge">'
        cmd = 'undo info-center source %s channel %s log' % (self.module_name, self.channel_id)
    xml_str += '<moduleName>%s</moduleName><icChannelId>%s</icChannelId>' % (self.module_name, self.channel_id)
    if self.log_enable != 'no_use':
        if self.state == 'present' and (not source or self.log_enable != source.get('logEnFlg')):
            xml_str += '<logEnFlg>%s</logEnFlg>' % self.log_enable
            if self.log_enable == 'true':
                cmd += ' state on'
            else:
                cmd += ' state off'
        elif self.state == 'absent' and source and (self.log_level == source.get('logEnLevel')):
            xml_str += '<logEnFlg>%s</logEnFlg>' % CHANNEL_DEFAULT_LOG_STATE.get(self.channel_id)
            cmd += ' state'
    if self.log_level:
        if self.state == 'present' and (not source or self.log_level != source.get('logEnLevel')):
            xml_str += '<logEnLevel>%s</logEnLevel>' % self.log_level
            cmd += ' level %s' % self.log_level
        elif self.state == 'absent' and source and (self.log_level == source.get('logEnLevel')):
            xml_str += '<logEnLevel>%s</logEnLevel>' % CHANNEL_DEFAULT_LOG_LEVEL.get(self.channel_id)
            cmd += ' level'
    if xml_str.endswith('</icChannelId>'):
        if self.log_enable == 'no_use' and (not self.log_level) and (self.state == 'absent'):
            xml_str += '</icSource></icSources>'
            self.updates_cmd.append(cmd)
            return xml_str
        else:
            return ''
    else:
        xml_str += '</icSource></icSources>'
        self.updates_cmd.append(cmd)
        return xml_str