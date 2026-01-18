from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_cluster_action(self, cluster_identity):
    cluster_action = None
    if self.parameters.get('cluster_name') is not None:
        cluster_action = self.na_helper.get_cd_action(cluster_identity, self.parameters)
        if cluster_action == 'delete':
            cluster_action = None
            self.na_helper.changed = False
    return cluster_action