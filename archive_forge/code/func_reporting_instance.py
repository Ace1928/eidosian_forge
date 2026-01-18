from pprint import pformat
from six import iteritems
import re
@reporting_instance.setter
def reporting_instance(self, reporting_instance):
    """
        Sets the reporting_instance of this V1beta1Event.
        ID of the controller instance, e.g. `kubelet-xyzf`.

        :param reporting_instance: The reporting_instance of this V1beta1Event.
        :type: str
        """
    self._reporting_instance = reporting_instance