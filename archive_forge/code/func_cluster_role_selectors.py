from pprint import pformat
from six import iteritems
import re
@cluster_role_selectors.setter
def cluster_role_selectors(self, cluster_role_selectors):
    """
        Sets the cluster_role_selectors of this V1alpha1AggregationRule.
        ClusterRoleSelectors holds a list of selectors which will be used to
        find ClusterRoles and create the rules. If any of the selectors match,
        then the ClusterRole's permissions will be added

        :param cluster_role_selectors: The cluster_role_selectors of this
        V1alpha1AggregationRule.
        :type: list[V1LabelSelector]
        """
    self._cluster_role_selectors = cluster_role_selectors