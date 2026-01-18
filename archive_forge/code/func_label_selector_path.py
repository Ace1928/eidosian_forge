from pprint import pformat
from six import iteritems
import re
@label_selector_path.setter
def label_selector_path(self, label_selector_path):
    """
        Sets the label_selector_path of this
        V1beta1CustomResourceSubresourceScale.
        LabelSelectorPath defines the JSON path inside of a CustomResource that
        corresponds to Scale.Status.Selector. Only JSON paths without the array
        notation are allowed. Must be a JSON Path under .status. Must be set to
        work with HPA. If there is no value under the given path in the
        CustomResource, the status label selector value in the /scale
        subresource will default to the empty string.

        :param label_selector_path: The label_selector_path of this
        V1beta1CustomResourceSubresourceScale.
        :type: str
        """
    self._label_selector_path = label_selector_path