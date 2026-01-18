from pprint import pformat
from six import iteritems
import re
@extra.setter
def extra(self, extra):
    """
        Sets the extra of this V1UserInfo.
        Any additional information provided by the authenticator.

        :param extra: The extra of this V1UserInfo.
        :type: dict(str, list[str])
        """
    self._extra = extra