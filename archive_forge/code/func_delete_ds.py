from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flashblade.plugins.module_utils.purefb import (
def delete_ds(module, blade):
    """Delete Directory Service"""
    changed = True
    if not module.check_mode:
        dirserv = blade.directory_services.list_directory_services(names=[module.params['dstype']])
        try:
            if module.params['dstype'] == 'management':
                if dirserv.items[0].uris:
                    dir_service = DirectoryService(uris=[''], base_dn='', bind_user='', bind_password='', enabled=False)
                else:
                    changed = False
            elif module.params['dstype'] == 'smb':
                if dirserv.items[0].uris:
                    smb_attrs = {'join_ou': ''}
                    dir_service = DirectoryService(uris=[''], base_dn='', bind_user='', bind_password='', smb=smb_attrs, enabled=False)
                else:
                    changed = False
            elif module.params['dstype'] == 'nfs':
                if dirserv.items[0].uris:
                    dir_service = DirectoryService(uris=[''], base_dn='', bind_user='', bind_password='', enabled=False)
                elif dirserv.items[0].nfs.nis_domains:
                    nfs_attrs = {'nis_domains': [], 'nis_servers': []}
                    dir_service = DirectoryService(nfs=nfs_attrs, enabled=False)
                else:
                    changed = False
            if changed:
                blade.directory_services.update_directory_services(names=[module.params['dstype']], directory_service=dir_service)
        except Exception:
            module.fail_json(msg='Delete {0} Directory Service failed'.format(module.params['dstype']))
    module.exit_json(changed=changed)