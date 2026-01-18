from pprint import pformat
from six import iteritems
import re
@label_selector.setter
def label_selector(self, label_selector):
    """
        Sets the label_selector of this V1PodAffinityTerm.
        A label query over a set of resources, in this case pods.

        :param label_selector: The label_selector of this V1PodAffinityTerm.
        :type: V1LabelSelector
        """
    self._label_selector = label_selector