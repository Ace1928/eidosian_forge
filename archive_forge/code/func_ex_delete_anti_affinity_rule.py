import re
import sys
from libcloud.utils.py3 import ET, urlencode, basestring, ensure_string
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.compute.base import (
from libcloud.common.nttcis import (
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
def ex_delete_anti_affinity_rule(self, anti_affinity_rule):
    """
        Remove anti affinity rule

        :param anti_affinity_rule: The anti affinity rule to delete
        :type  anti_affinity_rule: :class:`NttCisAntiAffinityRule` or
                                   ``str``

        :rtype: ``bool``
        """
    rule_id = anti_affinity_rule
    update_node = ET.Element('deleteAntiAffinityRule', {'xmlns': TYPES_URN})
    update_node.set('id', rule_id)
    result = self.connection.request_with_orgId_api_2('server/deleteAntiAffinityRule', method='POST', data=ET.tostring(update_node)).object
    response_code = findtext(result, 'responseCode', TYPES_URN)
    return response_code in ['IN_PROGRESS', 'SUCCESS']