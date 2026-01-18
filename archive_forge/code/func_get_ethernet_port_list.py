from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def get_ethernet_port_list(self):
    """Get the list of ethernet ports on a given Unity storage system"""
    try:
        LOG.info('Getting ethernet ports list')
        ethernet_port = self.unity.get_ethernet_port()
        return result_list(ethernet_port)
    except Exception as e:
        msg = 'Get ethernet port list from unity array failed with error %s' % str(e)
        LOG.error(msg)
        self.module.fail_json(msg=msg)