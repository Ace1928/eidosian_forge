import re
import sys
from libcloud.utils.py3 import ET, urlencode, basestring, ensure_string
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.compute.base import (
from libcloud.common.nttcis import (
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
def ex_get_vlan(self, vlan_id):
    """
        Get a single VLAN, by it's identifier

        :param   vlan_id: The identifier of the VLAN
        :type    vlan_id: ``str``

        :return: an instance of `NttCisVlan`
        :rtype: :class:`NttCisVlan`
        """
    locations = self.list_locations()
    vlan = self.connection.request_with_orgId_api_2('network/vlan/%s' % vlan_id).object
    return self._to_vlan(vlan, locations)