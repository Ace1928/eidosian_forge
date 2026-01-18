from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import ansible_collections.netapp.elementsw.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.elementsw.plugins.module_utils.netapp_elementsw_module import NaElementSWModule
def get_admin_user(self):
    """
        Get the admin user object
        :return: the admin user object
        """
    admins_list = self.sfe.list_cluster_admins()
    for admin in admins_list.cluster_admins:
        if admin.username == self.element_username:
            return admin
    return None