from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def get_hosts_list(self):
    """ Get the list of hosts on a given Unity storage system """
    try:
        LOG.info('Getting hosts list ')
        hosts = self.unity.get_host()
        return result_list(hosts)
    except Exception as e:
        msg = 'Get hosts list from unity array failed with error %s' % str(e)
        LOG.error(msg)
        self.module.fail_json(msg=msg)