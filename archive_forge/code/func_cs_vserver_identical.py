from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.netscaler.netscaler import (
def cs_vserver_identical(client, module, csvserver_proxy):
    csvserver_list = csvserver.get_filtered(client, 'name:%s' % module.params['name'])
    diff_dict = csvserver_proxy.diff_object(csvserver_list[0])
    if len(diff_dict) == 0:
        return True
    else:
        return False