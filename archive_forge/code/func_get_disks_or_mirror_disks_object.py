from __future__ import absolute_import, division, print_function
import re
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
import ansible_collections.netapp.ontap.plugins.module_utils.rest_response_helpers as rrh
@staticmethod
def get_disks_or_mirror_disks_object(name, disks):
    """
        create ZAPI object for disks or mirror_disks
        """
    disks_obj = netapp_utils.zapi.NaElement(name)
    for disk in disks:
        disk_info_obj = netapp_utils.zapi.NaElement('disk-info')
        disk_info_obj.add_new_child('name', disk)
        disks_obj.add_child_elem(disk_info_obj)
    return disks_obj