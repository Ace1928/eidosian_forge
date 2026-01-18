from pprint import pformat
from six import iteritems
import re
@group_version.setter
def group_version(self, group_version):
    """
        Sets the group_version of this V1GroupVersionForDiscovery.
        groupVersion specifies the API group and version in the form
        "group/version"

        :param group_version: The group_version of this
        V1GroupVersionForDiscovery.
        :type: str
        """
    if group_version is None:
        raise ValueError('Invalid value for `group_version`, must not be `None`')
    self._group_version = group_version