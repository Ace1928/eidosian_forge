from __future__ import absolute_import, division, print_function
import re
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_elementsw_module import NaElementSWModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def check_elementsw_parameters(self, kind='source'):
    """
        Validate all ElementSW cluster parameters required for managing the SnapMirror relationship
        Validate if both source and destination paths are present
        Validate if source_path follows the required format
        Validate SVIP
        Validate if ElementSW volume exists
        :return: None
        """
    path = None
    if kind == 'destination':
        path = self.parameters.get('destination_path')
    elif kind == 'source':
        path = self.parameters.get('source_path')
    if path is None:
        self.module.fail_json(msg='Error: Missing required parameter %s_path for connection_type %s' % (kind, self.parameters['connection_type']))
    if NetAppONTAPSnapmirror.element_source_path_format_matches(path) is None:
        self.module.fail_json(msg='Error: invalid %s_path %s. If the path is a ElementSW cluster, the value should be of the format <Element_SVIP>:/lun/<Element_VOLUME_ID>' % (kind, path))
    elementsw_helper, elem = self.set_element_connection(kind)
    self.validate_elementsw_svip(path, elem)
    self.check_if_elementsw_volume_exists(path, elementsw_helper)