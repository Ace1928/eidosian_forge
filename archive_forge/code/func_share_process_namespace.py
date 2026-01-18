from pprint import pformat
from six import iteritems
import re
@share_process_namespace.setter
def share_process_namespace(self, share_process_namespace):
    """
        Sets the share_process_namespace of this V1PodSpec.
        Share a single process namespace between all of the containers in a pod.
        When this is set containers will be able to view and signal processes
        from other containers in the same pod, and the first process in each
        container will not be assigned PID 1. HostPID and ShareProcessNamespace
        cannot both be set. Optional: Default to false. This field is beta-level
        and may be disabled with the PodShareProcessNamespace feature.

        :param share_process_namespace: The share_process_namespace of this
        V1PodSpec.
        :type: bool
        """
    self._share_process_namespace = share_process_namespace