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
def create_lun_path_rest(self):
    """ ZAPI accepts just a name, while REST expects a path. We need to convert a name in to a path for backward compatibility
            If the name start with a slash we will assume it a path and use it as the name
        """
    if not self.parameters['name'].startswith('/') and self.parameters.get('flexvol_name') is not None:
        if self.parameters.get('qtree_name') is not None:
            return '/vol/%s/%s/%s' % (self.parameters['flexvol_name'], self.parameters['qtree_name'], self.parameters['name'])
        return '/vol/%s/%s' % (self.parameters['flexvol_name'], self.parameters['name'])
    return self.parameters['name']