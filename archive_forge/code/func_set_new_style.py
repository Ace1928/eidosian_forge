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
def set_new_style(self):
    if not self.parameters.get('destination_endpoint') or not self.parameters.get('source_endpoint'):
        self.module.fail_json(msg='Missing parameters: Source endpoint or Destination endpoint')
    self.parameters['source_endpoint'] = self.na_helper.filter_out_none_entries(self.parameters['source_endpoint'])
    self.parameters['destination_endpoint'] = self.na_helper.filter_out_none_entries(self.parameters['destination_endpoint'])
    ontap_97_options = ['cluster', 'ipspace']
    if self.too_old(9, 7) and any((x in self.parameters['source_endpoint'] for x in ontap_97_options)):
        self.module.fail_json(msg='Error: %s' % self.rest_api.options_require_ontap_version(ontap_97_options, version='9.7', use_rest=self.use_rest))
    if self.too_old(9, 7) and any((x in self.parameters['destination_endpoint'] for x in ontap_97_options)):
        self.module.fail_json(msg='Error: %s' % self.rest_api.options_require_ontap_version(ontap_97_options, version='9.7', use_rest=self.use_rest))
    ontap_98_options = ['consistency_group_volumes']
    if self.too_old(9, 8) and any((x in self.parameters['source_endpoint'] for x in ontap_98_options)):
        self.module.fail_json(msg='Error: %s' % self.rest_api.options_require_ontap_version(ontap_98_options, version='9.8', use_rest=self.use_rest))
    if self.too_old(9, 8) and any((x in self.parameters['destination_endpoint'] for x in ontap_98_options)):
        self.module.fail_json(msg='Error: %s' % self.rest_api.options_require_ontap_version(ontap_98_options, version='9.8', use_rest=self.use_rest))
    self.parameters['source_cluster'] = self.na_helper.safe_get(self.parameters, ['source_endpoint', 'cluster'])
    self.parameters['source_path'] = self.na_helper.safe_get(self.parameters, ['source_endpoint', 'path'])
    self.parameters['source_vserver'] = self.na_helper.safe_get(self.parameters, ['source_endpoint', 'svm'])
    self.parameters['destination_cluster'] = self.na_helper.safe_get(self.parameters, ['destination_endpoint', 'cluster'])
    self.parameters['destination_path'] = self.na_helper.safe_get(self.parameters, ['destination_endpoint', 'path'])
    self.parameters['destination_vserver'] = self.na_helper.safe_get(self.parameters, ['destination_endpoint', 'svm'])
    self.new_style = True