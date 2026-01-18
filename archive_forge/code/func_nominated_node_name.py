from pprint import pformat
from six import iteritems
import re
@nominated_node_name.setter
def nominated_node_name(self, nominated_node_name):
    """
        Sets the nominated_node_name of this V1PodStatus.
        nominatedNodeName is set only when this pod preempts other pods on the
        node, but it cannot be scheduled right away as preemption victims
        receive their graceful termination periods. This field does not
        guarantee that the pod will be scheduled on this node. Scheduler may
        decide to place the pod elsewhere if other nodes become available
        sooner. Scheduler may also decide to give the resources on this node to
        a higher priority pod that is created after preemption. As a result,
        this field may be different than PodSpec.nodeName when the pod is
        scheduled.

        :param nominated_node_name: The nominated_node_name of this V1PodStatus.
        :type: str
        """
    self._nominated_node_name = nominated_node_name