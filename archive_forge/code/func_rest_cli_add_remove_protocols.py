from __future__ import absolute_import, division, print_function
import copy
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic, rest_vserver, zapis_svm
def rest_cli_add_remove_protocols(self, protocols):
    protocols_to_add = [protocol for protocol, value in protocols.items() if value]
    if protocols_to_add:
        self.rest_cli_add_protocols(protocols_to_add)
    protocols_to_delete = [protocol for protocol, value in protocols.items() if not value]
    if protocols_to_delete:
        self.rest_cli_remove_protocols(protocols_to_delete)