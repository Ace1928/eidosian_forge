import re
import sys
from libcloud.utils.py3 import ET, urlencode, basestring, ensure_string
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.compute.base import (
from libcloud.common.nttcis import (
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
def ex_get_image_by_id(self, id):
    """
        Gets a Base/Customer image in the NTTC-CIS Cloud given the id

        Note: This first checks the base image
              If it is not a base image we check if it is a customer image
              If it is not in either of these a NttCisAPIException
              is thrown

        :param id: The id of the image
        :type  id: ``str``

        :rtype: :class:`NodeImage`
        """
    try:
        return self.ex_get_base_image_by_id(id)
    except NttCisAPIException as e:
        if e.code != 'RESOURCE_NOT_FOUND':
            raise e
    return self.ex_get_customer_image_by_id(id)