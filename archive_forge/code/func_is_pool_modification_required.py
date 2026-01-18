from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def is_pool_modification_required(self, storage_pool_details):
    """ Check if attributes of storage pool needs to be modified
        """
    try:
        if self.module.params['new_pool_name'] and self.module.params['new_pool_name'] != storage_pool_details['name']:
            return True
        if self.module.params['pool_description'] is not None and self.module.params['pool_description'] != storage_pool_details['description']:
            return True
        if self.module.params['fast_cache']:
            if self.module.params['fast_cache'] == 'enabled' and (not storage_pool_details['is_fast_cache_enabled']) or (self.module.params['fast_cache'] == 'disabled' and storage_pool_details['is_fast_cache_enabled']):
                return True
        if self.module.params['fast_vp']:
            if self.module.params['fast_vp'] == 'enabled' and (not storage_pool_details['is_fast_vp_enabled']) or (self.module.params['fast_vp'] == 'disabled' and storage_pool_details['is_fast_vp_enabled']):
                return True
        LOG.info('modify not required')
        return False
    except Exception as e:
        error_message = 'Failed to determine if any modificationrequired for pool attributes with error: {0}'.format(str(e))
        LOG.error(error_message)
        self.module.fail_json(msg=error_message)