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
def modify_san_application(self, modify):
    """Use REST application/applications san template to add one or more LUNs"""
    body, error = self.create_san_app_body(modify)
    self.fail_on_error(error)
    body.pop('name')
    body.pop('svm')
    body.pop('smart_container')
    dummy, error = self.rest_app.patch_application(body)
    self.fail_on_error(error)