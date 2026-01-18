from pprint import pformat
from six import iteritems
import re
@target_selector.setter
def target_selector(self, target_selector):
    """
        Sets the target_selector of this V1beta2ScaleStatus.
        label selector for pods that should match the replicas count. This is a
        serializated version of both map-based and more expressive set-based
        selectors. This is done to avoid introspection in the clients. The
        string will be in the same format as the query-param syntax. If the
        target type only supports map-based selectors, both this field and
        map-based selector field are populated. More info:
        https://kubernetes.io/docs/concepts/overview/working-with-objects/labels/#label-selectors

        :param target_selector: The target_selector of this V1beta2ScaleStatus.
        :type: str
        """
    self._target_selector = target_selector