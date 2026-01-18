from __future__ import absolute_import, division, print_function
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config
def config_session(self):
    """configures isis"""
    xml_str = ''
    instance = self.isis_dict['instance']
    if not self.instance_id:
        return xml_str
    if self.state == 'present':
        xml_str = '<instanceId>%s</instanceId>' % self.instance_id
        self.updates_cmd.append('isis %s' % self.instance_id)
        if self.vpn_name:
            xml_str += '<vpnName>%s</vpnName>' % self.vpn_name
            self.updates_cmd.append('vpn-instance %s' % self.vpn_name)
    elif self.instance_id and str(self.instance_id) == instance.get('instanceId'):
        xml_str = '<instanceId>%s</instanceId>' % self.instance_id
        self.updates_cmd.append('undo isis %s' % self.instance_id)
    if self.state == 'present':
        return '<isSites><isSite operation="merge">' + xml_str + '</isSite></isSites>'
    elif xml_str:
        return '<isSites><isSite operation="delete">' + xml_str + '</isSite></isSites>'