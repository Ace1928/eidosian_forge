from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
def delete_ds_v6(module, array):
    """Delete Directory Service"""
    changed = True
    if module.params['dstype'] == 'management':
        management = flasharray.DirectoryServiceManagement(user_login_attribute='', user_object_class='')
        directory_service = flasharray.DirectoryService(uris=[''], base_dn='', bind_user='', bind_password='', enabled=False, services=module.params['dstype'], management=management)
    else:
        directory_service = flasharray.DirectoryService(uris=[''], base_dn='', bind_user='', bind_password='', enabled=False, services=module.params['dstype'])
    if not module.check_mode:
        res = array.patch_directory_services(names=[module.params['dstype']], directory_service=directory_service)
        if res.status_code != 200:
            module.fail_json(msg='Delete {0} Directory Service failed. Error message: {1}'.format(module.params['dstype'], res.errors[0].message))
    module.exit_json(changed=changed)