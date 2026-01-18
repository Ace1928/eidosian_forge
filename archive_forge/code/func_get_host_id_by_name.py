from __future__ import absolute_import, division, print_function
import logging
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def get_host_id_by_name(self, host_name):
    """ Get host ID by host name
        :param host_name: str
        :return: unity host ID
        :rtype: str
        """
    try:
        host_obj = self.unity_conn.get_host(name=host_name)
        if host_obj and host_obj.existed:
            return host_obj.id
        else:
            msg = 'Host name: %s does not exists' % host_name
            LOG.error(msg)
            self.module.fail_json(msg=msg)
    except Exception as e:
        msg = 'Failed to get host ID by name: %s error: %s' % (host_name, str(e))
        LOG.error(msg)
        self.module.fail_json(msg=msg)