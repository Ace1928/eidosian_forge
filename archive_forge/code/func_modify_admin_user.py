from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import ansible_collections.netapp.elementsw.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.elementsw.plugins.module_utils.netapp_elementsw_module import NaElementSWModule
def modify_admin_user(self):
    """
        Modify a admin user. If a password is set the user will be modified as there is no way to
        compare a new password with an existing one
        :return: if a user was modified or not
        """
    changed = False
    admin_user = self.get_admin_user()
    if self.access is not None and len(self.access) > 0:
        for access in self.access:
            if access not in admin_user.access:
                changed = True
    if changed and (not self.module.check_mode):
        self.sfe.modify_cluster_admin(cluster_admin_id=admin_user.cluster_admin_id, access=self.access, password=self.element_password, attributes=self.attributes)
    return changed