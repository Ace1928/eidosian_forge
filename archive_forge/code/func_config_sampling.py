from __future__ import (absolute_import, division, print_function)
import re
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec, check_ip_addr
def config_sampling(self):
    """configure sflow sampling on an interface"""
    xml_str = ''
    if not self.sflow_interface:
        return xml_str
    if not self.sflow_dict['sampling'] and self.state == 'absent':
        return xml_str
    self.updates_cmd.append('interface %s' % self.sflow_interface)
    if self.state == 'present':
        xml_str += '<samplings><sampling operation="merge"><ifName>%s</ifName>' % self.sflow_interface
    else:
        xml_str += '<samplings><sampling operation="delete"><ifName>%s</ifName>' % self.sflow_interface
    if self.sample_collector:
        if self.sflow_dict['sampling'].get('collectorID') and self.sflow_dict['sampling'].get('collectorID') != 'invalid':
            existing = self.sflow_dict['sampling'].get('collectorID').split(',')
        else:
            existing = list()
        if self.state == 'present':
            diff = list(set(self.sample_collector) - set(existing))
            if diff:
                self.updates_cmd.append('sflow sampling collector %s' % ' '.join(diff))
                new_set = list(self.sample_collector + existing)
                xml_str += '<collectorID>%s</collectorID>' % ','.join(list(set(new_set)))
        else:
            same = list(set(self.sample_collector) & set(existing))
            if same:
                self.updates_cmd.append('undo sflow sampling collector %s' % ' '.join(same))
                xml_str += '<collectorID>%s</collectorID>' % ','.join(list(set(same)))
    if self.sample_rate:
        exist = bool(self.sample_rate == self.sflow_dict['sampling'].get('rate'))
        if self.state == 'present' and (not exist):
            self.updates_cmd.append('sflow sampling rate %s' % self.sample_rate)
            xml_str += '<rate>%s</rate>' % self.sample_rate
        elif self.state == 'absent' and exist:
            self.updates_cmd.append('undo sflow sampling rate %s' % self.sample_rate)
            xml_str += '<rate>%s</rate>' % self.sample_rate
    if self.sample_length:
        exist = bool(self.sample_length == self.sflow_dict['sampling'].get('length'))
        if self.state == 'present' and (not exist):
            self.updates_cmd.append('sflow sampling length %s' % self.sample_length)
            xml_str += '<length>%s</length>' % self.sample_length
        elif self.state == 'absent' and exist:
            self.updates_cmd.append('undo sflow sampling length %s' % self.sample_length)
            xml_str += '<length>%s</length>' % self.sample_length
    if self.sample_direction:
        direction = list()
        if self.sample_direction == 'both':
            direction = ['inbound', 'outbound']
        else:
            direction.append(self.sample_direction)
        existing = list()
        if self.sflow_dict['sampling'].get('direction'):
            if self.sflow_dict['sampling'].get('direction') == 'both':
                existing = ['inbound', 'outbound']
            else:
                existing.append(self.sflow_dict['sampling'].get('direction'))
        if self.state == 'present':
            diff = list(set(direction) - set(existing))
            if diff:
                new_set = list(set(direction + existing))
                self.updates_cmd.append('sflow sampling %s' % ' '.join(diff))
                if len(new_set) > 1:
                    new_dir = 'both'
                else:
                    new_dir = new_set[0]
                xml_str += '<direction>%s</direction>' % new_dir
        else:
            same = list(set(existing) & set(direction))
            if same:
                self.updates_cmd.append('undo sflow sampling %s' % ' '.join(same))
                if len(same) > 1:
                    del_dir = 'both'
                else:
                    del_dir = same[0]
                xml_str += '<direction>%s</direction>' % del_dir
    if xml_str.endswith('</ifName>'):
        self.updates_cmd.pop()
        return ''
    xml_str += '</sampling></samplings>'
    return xml_str