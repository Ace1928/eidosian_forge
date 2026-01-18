from pprint import pformat
from six import iteritems
import re
@container_statuses.setter
def container_statuses(self, container_statuses):
    """
        Sets the container_statuses of this V1PodStatus.
        The list has one entry per container in the manifest. Each entry is
        currently the output of `docker inspect`. More info:
        https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle#pod-and-container-status

        :param container_statuses: The container_statuses of this V1PodStatus.
        :type: list[V1ContainerStatus]
        """
    self._container_statuses = container_statuses