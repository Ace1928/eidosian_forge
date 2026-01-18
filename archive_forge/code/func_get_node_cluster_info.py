from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.elementsw.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.elementsw.plugins.module_utils.netapp_elementsw_module import NaElementSWModule
def get_node_cluster_info(self):
    """
        Get Cluster Info - using node API
        """
    try:
        info = self.sfe_node.get_config()
        self.debug.append(repr(info.config.cluster))
        return info.config.cluster
    except Exception as exc:
        self.debug.append('port: %s, %s' % (str(self.sfe_node._port), repr(exc)))
        return None