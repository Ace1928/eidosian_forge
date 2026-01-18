import re
import sys
from libcloud.utils.py3 import ET, urlencode, basestring, ensure_string
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.compute.base import (
from libcloud.common.nttcis import (
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
def ex_set_firewall_rule_state(self, rule, state):
    """
        Change the state (enabled or disabled) of a rule

        :param rule: The rule to delete
        :type  rule: :class:`NttCisFirewallRule`

        :param state: The desired state enabled (True) or disabled (False)
        :type  state: ``bool``

        :rtype: ``bool``
        """
    update_node = ET.Element('editFirewallRule', {'xmlns': TYPES_URN})
    update_node.set('id', rule.id)
    ET.SubElement(update_node, 'enabled').text = str(state).lower()
    result = self.connection.request_with_orgId_api_2('network/editFirewallRule', method='POST', data=ET.tostring(update_node)).object
    response_code = findtext(result, 'responseCode', TYPES_URN)
    return response_code in ['IN_PROGRESS', 'OK']