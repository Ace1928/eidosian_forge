import re
import sys
from libcloud.utils.py3 import ET, urlencode, basestring, ensure_string
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.compute.base import (
from libcloud.common.nttcis import (
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
def ex_list_network_domains(self, location=None, name=None, service_plan=None, state=None):
    """
        List networks domains deployed across all data center locations domain.

        for your organization.
        The response includes the location of each network
        :param      location: Only network domains in the location (optional)
        :type       location: :class:`NodeLocation` or ``str``

        :param      name: Only network domains of this name (optional)
        :type       name: ``str``

        :param      service_plan: Only network domains of this type (optional)
        :type       service_plan: ``str``

        :param      state: Only network domains in this state (optional)
        :type       state: ``str``

        :return: a list of `NttCisNetwork` objects
        :rtype: ``list`` of :class:`NttCisNetwork`
        """
    params = {}
    if location is not None:
        params['datacenterId'] = self._location_to_location_id(location)
    if name is not None:
        params['name'] = name
    if service_plan is not None:
        params['type'] = service_plan
    if state is not None:
        params['state'] = state
    response = self.connection.request_with_orgId_api_2('network/networkDomain', params=params).object
    return self._to_network_domains(response)