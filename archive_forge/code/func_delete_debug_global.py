from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def delete_debug_global(self):
    """ Delete debug global """
    conf_str = CE_MERGE_DEBUG_GLOBAL_HEADER
    if self.debug_time_stamp:
        conf_str += '<debugTimeStamp>DATE_MILLISECOND</debugTimeStamp>'
    conf_str += CE_MERGE_DEBUG_GLOBAL_TAIL
    recv_xml = set_nc_config(self.module, conf_str)
    if '<ok/>' not in recv_xml:
        self.module.fail_json(msg='Error: delete debug global failed.')
    if self.debug_time_stamp:
        cmd = 'undo info-center timestamp debugging'
        self.updates_cmd.append(cmd)
    self.changed = True