from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell \
import copy
def set_mdm_virtual_interface(self, mdm_id=None, mdm_name=None, virtual_ip_interfaces=None, clear_interfaces=None, mdm_cluster_details=None):
    """Modify the MDM virtual IP interface.
        :param mdm_id: ID of MDM
        :param mdm_name: Name of MDM
        :param virtual_ip_interfaces: List of virtual IP interfaces
        :param clear_interfaces: clear virtual IP interfaces of MDM.
        :param mdm_cluster_details: Details of MDM cluster
        :return: True if modification of virtual interface or clear operation
                 successful
        """
    name_or_id = mdm_id if mdm_id else mdm_name
    if mdm_name is None and mdm_id is None:
        err_msg = 'Please provide mdm_name/mdm_id to modify virtual IP interfaces the MDM.'
        LOG.error(err_msg)
        self.module.fail_json(msg=err_msg)
    mdm_details = self.is_mdm_name_id_exists(mdm_name=mdm_name, mdm_id=mdm_id, cluster_details=mdm_cluster_details)
    if mdm_details is None:
        err_msg = self.not_exist_msg.format(name_or_id)
        self.module.fail_json(msg=err_msg)
    mdm_id = mdm_details['id']
    modify_list = []
    modify_list, clear = is_modify_mdm_virtual_interface(virtual_ip_interfaces, clear_interfaces, mdm_details)
    if modify_list is None and (not clear):
        LOG.info('No change required in MDM virtual IP interfaces.')
        return False
    try:
        log_msg = 'Modifying MDM virtual interfaces to %s or %s' % (str(modify_list), clear)
        LOG.debug(log_msg)
        if not self.module.check_mode:
            self.powerflex_conn.system.modify_virtual_ip_interface(mdm_id=mdm_id, virtual_ip_interfaces=modify_list, clear_interfaces=clear)
        return True
    except Exception as e:
        error_msg = 'Failed to modify the virtual IP interfaces of MDM {0} with error {1}'.format(name_or_id, str(e))
        LOG.error(error_msg)
        self.module.fail_json(msg=error_msg)