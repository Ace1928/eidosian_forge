from pprint import pformat
from six import iteritems
import re
@init_container_statuses.setter
def init_container_statuses(self, init_container_statuses):
    """
        Sets the init_container_statuses of this V1PodStatus.
        The list has one entry per init container in the manifest. The most
        recent successful init container will have ready = true, the most
        recently started container will have startTime set. More info:
        https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle#pod-and-container-status

        :param init_container_statuses: The init_container_statuses of this
        V1PodStatus.
        :type: list[V1ContainerStatus]
        """
    self._init_container_statuses = init_container_statuses