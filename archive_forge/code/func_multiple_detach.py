from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def multiple_detach(self, host_list_detach, obj_vol):
    """Detach multiple hosts from a volume
        :param host_list_detach: hosts to unmap the volume
        :param obj_vol: volume instance
        :return: response from API call
        """
    try:
        host_access = []
        for item in host_list_detach:
            host_access.append({'accessMask': utils.HostLUNAccessEnum.PRODUCTION, 'host': {'id': item}})
        resp = obj_vol.modify(host_access=host_access)
        return resp
    except Exception as e:
        errormsg = 'Failed to detach hosts {0} from volume {1} with error {2} '.format(host_list_detach, obj_vol.name, str(e))
        LOG.error(errormsg)
        self.module.fail_json(msg=errormsg)