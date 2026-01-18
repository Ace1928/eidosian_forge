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
def adjust_sizes(self, current, after_create):
    """
        ignore small change in size by resetting expectations
        """
    if after_create:
        self.parameters['size'] = current['size']
        return
    self.ignore_small_change(current, 'size', self.parameters['size_change_threshold'])
    self.ignore_small_change(current, 'max_files', netapp_utils.get_feature(self.module, 'max_files_change_threshold'))