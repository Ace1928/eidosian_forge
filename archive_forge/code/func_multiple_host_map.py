from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def multiple_host_map(self, host_dic_list, obj_vol):
    """Attach multiple hosts to a volume
        :param host_dic_list: hosts to map the volume
        :param obj_vol: volume instance
        :return: response from API call
        """
    try:
        host_access = []
        current_hosts = self.get_volume_host_access_list(obj_vol)
        for existing_host in current_hosts:
            host_access.append({'accessMask': eval('utils.HostLUNAccessEnum.' + existing_host['accessMask']), 'host': {'id': existing_host['id']}, 'hlu': existing_host['hlu']})
        for item in host_dic_list:
            host_access.append({'accessMask': utils.HostLUNAccessEnum.PRODUCTION, 'host': {'id': item['host_id']}, 'hlu': item['hlu']})
        resp = obj_vol.modify(host_access=host_access)
        return resp
    except Exception as e:
        errormsg = 'Failed to attach hosts {0} with volume {1} with error {2} '.format(host_dic_list, obj_vol.name, str(e))
        LOG.error(errormsg)
        self.module.fail_json(msg=errormsg)