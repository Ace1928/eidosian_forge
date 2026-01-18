import re
import sys
from libcloud.utils.py3 import ET, urlencode, basestring, ensure_string
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.compute.base import (
from libcloud.common.nttcis import (
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
def ex_get_tag_key_by_name(self, name):
    """
        NOTICE: Tag key is one of those instances where Libloud
                handles the search of a list for the client code.
                This behavior exists inconsistently across libcloud.
                Get a specific tag key by Name

        :param name: Name of the tag key you want (required)
        :type  name: ``str``

        :rtype: :class:`NttCisTagKey`
        """
    tag_keys = self.ex_list_tag_keys(name=name)
    if len(tag_keys) != 1:
        raise ValueError('No tags found with name %s' % name)
    return tag_keys[0]