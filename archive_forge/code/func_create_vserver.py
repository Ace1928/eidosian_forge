from __future__ import absolute_import, division, print_function
import copy
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic, rest_vserver, zapis_svm
def create_vserver(self):
    if self.use_rest:
        self.create_vserver_rest()
    else:
        options = {'vserver-name': self.parameters['name']}
        self.add_parameter_to_dict(options, 'root_volume', 'root-volume')
        self.add_parameter_to_dict(options, 'root_volume_aggregate', 'root-volume-aggregate')
        self.add_parameter_to_dict(options, 'root_volume_security_style', 'root-volume-security-style')
        self.add_parameter_to_dict(options, 'language', 'language')
        self.add_parameter_to_dict(options, 'ipspace', 'ipspace')
        self.add_parameter_to_dict(options, 'snapshot_policy', 'snapshot-policy')
        self.add_parameter_to_dict(options, 'subtype', 'vserver-subtype')
        self.add_parameter_to_dict(options, 'comment', 'comment')
        vserver_create = netapp_utils.zapi.NaElement.create_node_with_children('vserver-create', **options)
        try:
            self.server.invoke_successfully(vserver_create, enable_tunneling=False)
        except netapp_utils.zapi.NaApiError as exc:
            self.module.fail_json(msg='Error provisioning SVM %s: %s' % (self.parameters['name'], to_native(exc)), exception=traceback.format_exc())
        options = dict(((key, self.parameters[key]) for key in ('allowed_protocols', 'aggr_list', 'max_volumes') if self.parameters.get(key)))
        if options:
            self.modify_vserver(options)