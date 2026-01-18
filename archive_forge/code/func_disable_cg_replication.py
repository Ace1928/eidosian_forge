from __future__ import absolute_import, division, print_function
import logging
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def disable_cg_replication(self, cg_name):
    """ Remove replication from the consistency group """
    try:
        cg_object = self.return_cg_instance(cg_name)
        if not cg_object.check_cg_is_replicated():
            return False
        LOG.info(('Disabling replication from the consistency group %s', cg_object.name))
        curr_cg_repl_session = self.unity_conn.get_replication_session(src_resource_id=cg_object.id)
        for repl_session in curr_cg_repl_session:
            repl_session.delete()
        return True
    except Exception as e:
        errormsg = 'Disabling replication to the consistency group %s failed with error %s' % (cg_object.name, str(e))
        LOG.error(errormsg)
        self.module.fail_json(msg=errormsg)