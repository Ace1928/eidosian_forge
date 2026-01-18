from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic, rest_vserver
def sanitize_acl_for_post(self, acl):
    """ some fields like access_control, propagation_mode are not accepted for POST operation """
    post_acl = dict(acl)
    for key in acl:
        if key not in self.post_acl_keys:
            post_acl.pop(key)
    return post_acl