from pprint import pformat
from six import iteritems
import re
@api_groups.setter
def api_groups(self, api_groups):
    """
        Sets the api_groups of this V1beta1ResourceRule.
        APIGroups is the name of the APIGroup that contains the resources.  If
        multiple API groups are specified, any action requested against one of
        the enumerated resources in any API group will be allowed.  "*" means
        all.

        :param api_groups: The api_groups of this V1beta1ResourceRule.
        :type: list[str]
        """
    self._api_groups = api_groups