from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def create_cifs_share(self):
    """
        Create CIFS share
        """
    options = {'share-name': self.parameters.get('name'), 'path': self.parameters.get('path')}
    cifs_create = netapp_utils.zapi.NaElement.create_node_with_children('cifs-share-create', **options)
    self.create_modify_cifs_share(cifs_create, 'creating')