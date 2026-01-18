from pprint import pformat
from six import iteritems
import re
@number_misscheduled.setter
def number_misscheduled(self, number_misscheduled):
    """
        Sets the number_misscheduled of this V1beta2DaemonSetStatus.
        The number of nodes that are running the daemon pod, but are not
        supposed to run the daemon pod. More info:
        https://kubernetes.io/docs/concepts/workloads/controllers/daemonset/

        :param number_misscheduled: The number_misscheduled of this
        V1beta2DaemonSetStatus.
        :type: int
        """
    if number_misscheduled is None:
        raise ValueError('Invalid value for `number_misscheduled`, must not be `None`')
    self._number_misscheduled = number_misscheduled