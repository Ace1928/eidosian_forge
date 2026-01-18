from pprint import pformat
from six import iteritems
import re
@api_group.setter
def api_group(self, api_group):
    """
        Sets the api_group of this V1beta1RoleRef.
        APIGroup is the group for the resource being referenced

        :param api_group: The api_group of this V1beta1RoleRef.
        :type: str
        """
    if api_group is None:
        raise ValueError('Invalid value for `api_group`, must not be `None`')
    self._api_group = api_group