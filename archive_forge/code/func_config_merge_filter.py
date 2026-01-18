from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import ce_argument_spec, get_nc_config, set_nc_config, check_ip_addr
def config_merge_filter(self, filter_feature_name, filter_log_name):
    """config filter"""
    if not self.is_exist_filter(filter_feature_name, filter_log_name):
        conf_str = CE_NC_CREATE_CHANNEL_FILTER_HEADER
        conf_str += '<icFilterFlag>true</icFilterFlag>'
        if filter_feature_name:
            conf_str += '<icFeatureName>%s</icFeatureName>' % filter_feature_name
        if filter_log_name:
            conf_str += '<icFilterLogName>%s</icFilterLogName>' % filter_log_name
        conf_str += CE_NC_CREATE_CHANNEL_FILTER_TAIL
        recv_xml = set_nc_config(self.module, conf_str)
        if '<ok/>' not in recv_xml:
            self.module.fail_json(msg='Error: Merge syslog filter failed.')
        self.updates_cmd.append('info-center filter-id bymodule-alias %s %s' % (filter_feature_name, filter_log_name))
        self.changed = True