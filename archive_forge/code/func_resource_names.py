from pprint import pformat
from six import iteritems
import re
@resource_names.setter
def resource_names(self, resource_names):
    """
        Sets the resource_names of this V1beta1ResourceRule.
        ResourceNames is an optional white list of names that the rule applies
        to.  An empty set means that everything is allowed.  "*" means all.

        :param resource_names: The resource_names of this V1beta1ResourceRule.
        :type: list[str]
        """
    self._resource_names = resource_names