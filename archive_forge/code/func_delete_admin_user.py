from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import ansible_collections.netapp.elementsw.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.elementsw.plugins.module_utils.netapp_elementsw_module import NaElementSWModule
def delete_admin_user(self):
    """
        Deletes an existing admin user from the element cluster
        :return: nothing
        """
    admin_user = self.get_admin_user()
    self.sfe.remove_cluster_admin(cluster_admin_id=admin_user.cluster_admin_id)