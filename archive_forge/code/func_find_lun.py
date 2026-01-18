from __future__ import absolute_import, division, print_function
import copy
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.rest_application import RestApplication
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_volume
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def find_lun(self, luns, name, lun_path=None):
    """
        Return lun record matching name or path

        :return: lun record
        :rtype: XML for ZAPI, dict for REST, or None if not found
        """
    if luns:
        for lun in luns:
            path = lun['path']
            if lun_path is None:
                if name == path:
                    return lun
                _rest, _splitter, found_name = path.rpartition('/')
                if found_name == name:
                    return lun
            elif lun_path == path:
                return lun
    return None