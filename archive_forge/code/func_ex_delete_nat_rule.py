import re
import sys
from libcloud.utils.py3 import ET, urlencode, basestring, ensure_string
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.compute.base import (
from libcloud.common.nttcis import (
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
def ex_delete_nat_rule(self, rule):
    """
        Delete an existing NAT rule

        :param  rule: The rule to delete
        :type   rule: :class:`NttCisNatRule`

        :rtype: ``bool``
        """
    update_node = ET.Element('deleteNatRule', {'xmlns': TYPES_URN})
    update_node.set('id', rule.id)
    result = self.connection.request_with_orgId_api_2('network/deleteNatRule', method='POST', data=ET.tostring(update_node)).object
    response_code = findtext(result, 'responseCode', TYPES_URN)
    return response_code in ['IN_PROGRESS', 'OK']