import re
import sys
from libcloud.utils.py3 import ET, urlencode, basestring, ensure_string
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.compute.base import (
from libcloud.common.nttcis import (
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
def ex_delete_portlist(self, ex_portlist):
    """
        Delete Port List

        >>> from pprint import pprint
        >>> from libcloud.compute.types import Provider
        >>> from libcloud.compute.providers import get_driver
        >>> import libcloud.security
        >>>
        >>> # Get NTTC-CIS driver
        >>> libcloud.security.VERIFY_SSL_CERT = True
        >>> cls = get_driver(Provider.NTTCIS)
        >>> driver = cls('myusername','mypassword', region='dd-au')
        >>>
        >>> # Delete Port List
        >>> portlist_id = '157531ce-77d4-493c-866b-d3d3fc4a912a'
        >>> response = driver.ex_delete_portlist(portlist_id)
        >>> pprint(response)

        :param    ex_portlist:  Port List to be deleted
        :type     ex_portlist: :``str`` or :class:'NttCisPortList'

        :rtype: ``bool``
        """
    delete_port_list = ET.Element('deletePortList', {'xmlns': TYPES_URN, 'id': self._port_list_to_port_list_id(ex_portlist)})
    response = self.connection.request_with_orgId_api_2('network/deletePortList', method='POST', data=ET.tostring(delete_port_list)).object
    response_code = findtext(response, 'responseCode', TYPES_URN)
    return response_code in ['IN_PROGRESS', 'OK']