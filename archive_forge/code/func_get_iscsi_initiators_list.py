from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def get_iscsi_initiators_list(self):
    """ Get the list of ISCSI initiators on a given Unity storage
            system """
    try:
        LOG.info('Getting ISCSI initiators list ')
        iscsi_initiator = utils.host.UnityHostInitiatorList.get(cli=self.unity._cli, type=utils.HostInitiatorTypeEnum.ISCSI)
        return iscsi_initiators_result_list(iscsi_initiator)
    except Exception as e:
        msg = 'Get ISCSI initiators list from unity array failed with error %s' % str(e)
        LOG.error(msg)
        self.module.fail_json(msg=msg)