from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def get_io_limit_policy(self, name=None, id=None):
    """Get the instance of a io limit policy.
            :param name: The io limit policy name
            :param id: The io limit policy id
            :return: instance of the respective io_limit_policy if exist.
        """
    errormsg = 'Failed to get the io limit policy {0} with error {1}'
    id_or_name = name if name else id
    try:
        obj_iopol = self.unity_conn.get_io_limit_policy(_id=id, name=name)
        if id and obj_iopol.existed:
            LOG.info('Successfully got the IO limit policy object %s', obj_iopol)
            return obj_iopol
        elif name:
            LOG.info('Successfully got the IO limit policy object %s ', obj_iopol)
            return obj_iopol
        else:
            msg = 'Failed to get the io limit policy with {0}'.format(id_or_name)
            LOG.error(msg)
            self.module.fail_json(msg=msg)
    except Exception as e:
        msg = errormsg.format(name, str(e))
        LOG.error(msg)
        self.module.fail_json(msg=msg)