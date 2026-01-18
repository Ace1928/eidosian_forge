from pprint import pformat
from six import iteritems
import re
@path_prefix.setter
def path_prefix(self, path_prefix):
    """
        Sets the path_prefix of this PolicyV1beta1AllowedHostPath.
        pathPrefix is the path prefix that the host volume must match. It does
        not support `*`. Trailing slashes are trimmed when validating the path
        prefix with a host path.  Examples: `/foo` would allow `/foo`, `/foo/`
        and `/foo/bar` `/foo` would not allow `/food` or `/etc/foo`

        :param path_prefix: The path_prefix of this
        PolicyV1beta1AllowedHostPath.
        :type: str
        """
    self._path_prefix = path_prefix