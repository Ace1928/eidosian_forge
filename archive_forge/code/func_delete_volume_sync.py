from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.rest_application import RestApplication
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
from ansible_collections.netapp.ontap.plugins.module_utils import rest_vserver
def delete_volume_sync(self, current, unmount_offline):
    """Delete ONTAP volume for flexvol types """
    options = {'name': self.parameters['name']}
    if unmount_offline:
        options['unmount-and-offline'] = 'true'
    volume_delete = netapp_utils.zapi.NaElement.create_node_with_children('volume-destroy', **options)
    try:
        self.server.invoke_successfully(volume_delete, enable_tunneling=True)
    except netapp_utils.zapi.NaApiError as error:
        return error
    return None