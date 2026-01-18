from __future__ import absolute_import, division, print_function
import re
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def create_nfs_share_from_filesystem(self):
    """ Create nfs share from given filesystem

        :return: nfs_share object
        :rtype: UnityNfsShare
        """
    name = self.module.params['nfs_export_name']
    path = self.module.params['path']
    if not name or not path:
        msg = 'Please provide name and path both for create'
        LOG.error(msg)
        self.module.fail_json(msg=msg)
    param = self.get_adv_param_from_pb()
    if 'default_access' in param:
        param['share_access'] = param.pop('default_access')
        LOG.info("Param name: 'share_access' is used instead of 'default_access' in SDK so changed")
    param = self.correct_payload_as_per_sdk(param)
    LOG.info('Creating nfs share from filesystem with param: %s', param)
    try:
        nfs_obj = utils.UnityNfsShare.create(cli=self.cli, name=name, fs=self.fs_obj, path=path, **param)
        LOG.info('Successfully created nfs share: %s', nfs_obj)
        return nfs_obj
    except utils.UnityNfsShareNameExistedError as e:
        LOG.error(str(e))
        self.module.fail_json(msg=str(e))
    except Exception as e:
        msg = 'Failed to create nfs share: %s error: %s' % (name, str(e))
        LOG.error(msg)
        self.module.fail_json(msg=msg)