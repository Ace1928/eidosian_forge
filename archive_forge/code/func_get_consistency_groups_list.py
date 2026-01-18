from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def get_consistency_groups_list(self):
    """ Get the list of consistency groups on a given Unity storage
            system """
    try:
        LOG.info('Getting consistency groups list ')
        consistency_groups = utils.cg.UnityConsistencyGroupList.get(self.unity._cli)
        return result_list(consistency_groups)
    except Exception as e:
        msg = 'Get consistency groups list from unity array failed with error %s' % str(e)
        LOG.error(msg)
        self.module.fail_json(msg=msg)