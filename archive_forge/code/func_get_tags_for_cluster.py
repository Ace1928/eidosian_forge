from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import env_fallback, missing_required_lib
from ansible.module_utils._text import to_native
def get_tags_for_cluster(self, cluster_mid=None):
    """
        Return list of tag object associated with cluster
        Args:
            cluster_mid: Dynamic object for cluster

        Returns: List of tag object associated with the given cluster

        """
    dobj = DynamicID(type='ClusterComputeResource', id=cluster_mid)
    return self.get_tags_for_dynamic_obj(dobj=dobj)