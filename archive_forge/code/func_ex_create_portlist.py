import re
import sys
from libcloud.utils.py3 import ET, urlencode, basestring, ensure_string
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.compute.base import (
from libcloud.common.nttcis import (
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
def ex_create_portlist(self, ex_network_domain, name, description, port_collection, child_portlist_list=None):
    """
        Create Port List.

        >>> from pprint import pprint
        >>> from libcloud.compute.types import Provider
        >>> from libcloud.compute.providers import get_driver
        >>> from libcloud.common.nttcis import NttCisPort
        >>> import libcloud.security
        >>>
        >>> # Get NTTC-CIS driver
        >>> libcloud.security.VERIFY_SSL_CERT = True
        >>> cls = get_driver(Provider.NTTCIS)
        >>> driver = cls('myusername','mypassword', region='dd-au')
        >>>
        >>> # Get location
        >>> location = driver.ex_get_location_by_id(id='AU9')
        >>>
        >>> # Get network domain by location
        >>> networkDomainName = "Baas QA"
        >>> network_domains = driver.ex_list_network_domains(location=location)
        >>> my_network_domain = [d for d in network_domains if d.name ==
                              networkDomainName][0]
        >>>
        >>> # Port Collection
        >>> port_1 = DimensionDataPort(begin='1000')
        >>> port_2 = DimensionDataPort(begin='1001', end='1003')
        >>> port_collection = [port_1, port_2]
        >>>
        >>> # Create Port List
        >>> new_portlist = driver.ex_create_portlist(
        >>>     ex_network_domain=my_network_domain,
        >>>     name='MyPortListX',
        >>>     description="Test only",
        >>>     port_collection=port_collection,
        >>>     child_portlist_list={'a9cd4984-6ff5-4f93-89ff-8618ab642bb9'}
        >>>     )
        >>> pprint(new_portlist)

        :param    ex_network_domain:  (required) The network domain in
                                       which to create PortList. Provide
                                       networkdomain object or its id.
        :type      ex_network_domain: :``str``

        :param    name:  Port List Name
        :type     name: :``str``

        :param    description:  IP Address List Description
        :type     description: :``str``

        :param    port_collection:  List of Port Address
        :type     port_collection: :``str``

        :param    child_portlist_list:  List of Child Portlist to be
                                        included in this Port List
        :type     child_portlist_list: :``str`` or ''list of
                                         :class:'NttCisChildPortList'

        :return: result of operation
        :rtype: ``bool``
        """
    new_port_list = ET.Element('createPortList', {'xmlns': TYPES_URN})
    ET.SubElement(new_port_list, 'networkDomainId').text = self._network_domain_to_network_domain_id(ex_network_domain)
    ET.SubElement(new_port_list, 'name').text = name
    ET.SubElement(new_port_list, 'description').text = description
    for port in port_collection:
        p = ET.SubElement(new_port_list, 'port')
        p.set('begin', port.begin)
        if port.end:
            p.set('end', port.end)
    if child_portlist_list is not None:
        for child in child_portlist_list:
            ET.SubElement(new_port_list, 'childPortListId').text = self._child_port_list_to_child_port_list_id(child)
    response = self.connection.request_with_orgId_api_2('network/createPortList', method='POST', data=ET.tostring(new_port_list)).object
    response_code = findtext(response, 'responseCode', TYPES_URN)
    return response_code in ['IN_PROGRESS', 'OK']