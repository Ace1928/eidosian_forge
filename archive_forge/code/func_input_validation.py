from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def input_validation(self):
    if self.parameters.get('vserver') == '*':
        self.module.fail_json(msg='As svm name * represents all svms and created by default, please provide a specific SVM name')
    if self.parameters.get('applications') == [''] and self.parameters.get('state') == 'present':
        self.module.fail_json(msg='Applications field cannot be empty, at least one application must be specified')