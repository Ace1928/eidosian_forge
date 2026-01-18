from pprint import pformat
from six import iteritems
import re
@go_version.setter
def go_version(self, go_version):
    """
        Sets the go_version of this VersionInfo.

        :param go_version: The go_version of this VersionInfo.
        :type: str
        """
    if go_version is None:
        raise ValueError('Invalid value for `go_version`, must not be `None`')
    self._go_version = go_version