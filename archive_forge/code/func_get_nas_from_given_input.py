from __future__ import absolute_import, division, print_function
import re
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def get_nas_from_given_input(self):
    """ Get nas server object

        :return: nas server object
        :rtype: UnityNasServer
        """
    LOG.info('Getting nas server details')
    if not self.module.params['nas_server_id'] and (not self.module.params['nas_server_name']):
        return None
    id_or_name = self.module.params['nas_server_id'] if self.module.params['nas_server_id'] else self.module.params['nas_server_name']
    try:
        nas = self.unity.get_nas_server(_id=self.module.params['nas_server_id'], name=self.module.params['nas_server_name'])
    except utils.UnityResourceNotFoundError as e:
        msg = 'Given nas server not found error: %s' % str(e)
        LOG.error(msg)
        self.module.fail_json(msg=msg)
    except utils.HTTPClientError as e:
        if e.http_status == 401:
            msg = 'Failed to get nas server: %s due to incorrect username/password error: %s' % (id_or_name, str(e))
        else:
            msg = 'Failed to get nas server: %s error: %s' % (id_or_name, str(e))
        LOG.error(msg)
        self.module.fail_json(msg=msg)
    except Exception as e:
        msg = 'Failed to get nas server: %s error: %s' % (id_or_name, str(e))
        LOG.error(msg)
        self.module.fail_json(msg=msg)
    if nas and (not nas.existed):
        msg = 'Please check nas details it does not exists'
        LOG.error(msg)
        self.module.fail_json(msg=msg)
    LOG.info('Got nas server details')
    return nas