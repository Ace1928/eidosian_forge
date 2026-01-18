from pprint import pformat
from six import iteritems
import re
@reporting_controller.setter
def reporting_controller(self, reporting_controller):
    """
        Sets the reporting_controller of this V1beta1Event.
        Name of the controller that emitted this Event, e.g.
        `kubernetes.io/kubelet`.

        :param reporting_controller: The reporting_controller of this
        V1beta1Event.
        :type: str
        """
    self._reporting_controller = reporting_controller