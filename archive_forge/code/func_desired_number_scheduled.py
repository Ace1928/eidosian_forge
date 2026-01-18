from pprint import pformat
from six import iteritems
import re
@desired_number_scheduled.setter
def desired_number_scheduled(self, desired_number_scheduled):
    """
        Sets the desired_number_scheduled of this V1beta2DaemonSetStatus.
        The total number of nodes that should be running the daemon pod
        (including nodes correctly running the daemon pod). More info:
        https://kubernetes.io/docs/concepts/workloads/controllers/daemonset/

        :param desired_number_scheduled: The desired_number_scheduled of this
        V1beta2DaemonSetStatus.
        :type: int
        """
    if desired_number_scheduled is None:
        raise ValueError('Invalid value for `desired_number_scheduled`, must not be `None`')
    self._desired_number_scheduled = desired_number_scheduled