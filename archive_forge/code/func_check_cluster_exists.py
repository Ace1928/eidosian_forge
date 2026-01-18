from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.elementsw.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.elementsw.plugins.module_utils.netapp_elementsw_module import NaElementSWModule
def check_cluster_exists(self):
    """
        validate if cluster exists with list of nodes
        error out if something is found but with different nodes
        return a tuple (found, info)
            found is True if found, False if not found
        """
    info = self.get_node_cluster_info()
    if info is None:
        return False
    ensemble = getattr(info, 'ensemble', None)
    if not ensemble:
        return False
    nodes = [x.split(':', 1)[1] for x in ensemble]
    current_ensemble_nodes = set(nodes) if ensemble else set()
    requested_nodes = set(self.nodes) if self.nodes else set()
    extra_ensemble_nodes = current_ensemble_nodes - requested_nodes
    if extra_ensemble_nodes and self.fail_if_cluster_already_exists_with_larger_ensemble:
        msg = 'Error: found existing cluster with more nodes in ensemble.  Cluster: %s, extra nodes: %s' % (getattr(info, 'cluster', 'not found'), extra_ensemble_nodes)
        msg += '.  Cluster info: %s' % repr(info)
        self.module.fail_json(msg=msg)
    if extra_ensemble_nodes:
        self.debug.append('Extra ensemble nodes: %s' % extra_ensemble_nodes)
    nodes_not_in_ensemble = requested_nodes - current_ensemble_nodes
    if nodes_not_in_ensemble:
        self.debug.append('Extra requested nodes not in ensemble: %s' % nodes_not_in_ensemble)
    return True