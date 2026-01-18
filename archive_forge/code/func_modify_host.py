from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
import ipaddress
def modify_host(self, host_details, new_host_name=None, description=None, host_os=None):
    """  Modify a host """
    try:
        hosts = utils.host.UnityHostList.get(cli=self.unity._cli)
        host_names_list = hosts.name
        for name in host_names_list:
            if new_host_name == name:
                error_message = 'Cannot modify name, new_host_name: ' + new_host_name + ' already in use.'
                LOG.error(error_message)
                self.module.fail_json(msg=error_message)
        host_details.modify(name=new_host_name, desc=description, os=host_os)
        return True
    except Exception as e:
        error_message = 'Got error %s while modifying host %s' % (str(e), host_details.name)
        LOG.error(error_message)
        self.module.fail_json(msg=error_message)