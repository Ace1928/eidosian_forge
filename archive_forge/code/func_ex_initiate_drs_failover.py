import re
import sys
from libcloud.utils.py3 import ET, urlencode, basestring, ensure_string
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.compute.base import (
from libcloud.common.nttcis import (
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
def ex_initiate_drs_failover(self, consistency_group_id):
    """
        This method is irreversible.
        It will failover the Consistency Group while removing it as well.

        :param consistency_group_id: Consistency Group's Id to failover
        :type consistency_group_id: ``str``

        :return: True if response_code contains either
        IN_PROGRESS' or 'OK' otherwise False
        :rtype: ``bool``
        """
    failover_elm = ET.Element('initiateFailover', {'consistencyGroupId': consistency_group_id, 'xmlns': TYPES_URN})
    response = self.connection.request_with_orgId_api_2('consistencyGroup/initiateFailover', method='POST', data=ET.tostring(failover_elm)).object
    response_code = findtext(response, 'responseCode', TYPES_URN)
    return response_code in ['IN_PROGRESS', 'OK']