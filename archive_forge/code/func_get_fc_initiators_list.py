from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def get_fc_initiators_list(self):
    """ Get the list of FC Initiators on a given Unity storage system """
    try:
        LOG.info('Getting FC initiators list ')
        fc_initiator = utils.host.UnityHostInitiatorList.get(cli=self.unity._cli, type=utils.HostInitiatorTypeEnum.FC)
        return fc_initiators_result_list(fc_initiator)
    except Exception as e:
        msg = 'Get FC initiators list from unity array failed with error %s' % str(e)
        LOG.error(msg)
        self.module.fail_json(msg=msg)