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
def get_lun_path_from_backend(self, name):
    """returns lun path matching name if found in backing_storage
           retruns None if not found
        """
    lun_paths = self.get_lun_paths_from_app()
    match = '/%s' % name
    return next((path for path in lun_paths if path.endswith(match)), None)