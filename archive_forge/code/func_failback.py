from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def failback(self, session_obj, force_full_copy):
    """Failback the replication session.
            :param session_obj: Replication session object
            :param force_full_copy: needed when replication session goes out of sync due to a fault.
            :return: True if failback is successful.
        """
    try:
        LOG.info('Failback replication session %s', session_obj.name)
        if session_obj.status.name in (utils.ReplicationOpStatusEnum.FAILED_OVER.name, utils.ReplicationOpStatusEnum.FAILED_OVER_WITH_SYNC.name, utils.ReplicationOpStatusEnum.PAUSED.name):
            if not self.module.check_mode:
                session_obj.failback(force_full_copy=force_full_copy)
            return True
    except Exception as e:
        msg = f'Failback replication session {session_obj.name} failed with error {str(e)}'
        LOG.error(msg)
        self.module.fail_json(msg=msg)