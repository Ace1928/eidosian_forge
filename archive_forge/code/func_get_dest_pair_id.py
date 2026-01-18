from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.elementsw.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.elementsw.plugins.module_utils.netapp_elementsw_module import NaElementSWModule
from ansible_collections.netapp.elementsw.plugins.module_utils.netapp_module import NetAppModule
def get_dest_pair_id(self):
    """
        Getting destination cluster_pair_id
        """
    paired_clusters = self.dest_elem.list_cluster_pairs()
    return self.check_if_already_paired(paired_clusters, self.parameters['hostname'])