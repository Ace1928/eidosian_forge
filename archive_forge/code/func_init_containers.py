from pprint import pformat
from six import iteritems
import re
@init_containers.setter
def init_containers(self, init_containers):
    """
        Sets the init_containers of this V1PodSpec.
        List of initialization containers belonging to the pod. Init containers
        are executed in order prior to containers being started. If any init
        container fails, the pod is considered to have failed and is handled
        according to its restartPolicy. The name for an init container or normal
        container must be unique among all containers. Init containers may not
        have Lifecycle actions, Readiness probes, or Liveness probes. The
        resourceRequirements of an init container are taken into account during
        scheduling by finding the highest request/limit for each resource type,
        and then using the max of of that value or the sum of the normal
        containers. Limits are applied to init containers in a similar fashion.
        Init containers cannot currently be added or removed. Cannot be updated.
        More info:
        https://kubernetes.io/docs/concepts/workloads/pods/init-containers/

        :param init_containers: The init_containers of this V1PodSpec.
        :type: list[V1Container]
        """
    self._init_containers = init_containers