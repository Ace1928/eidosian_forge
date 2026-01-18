import re
import sys
from libcloud.utils.py3 import ET, urlencode, basestring, ensure_string
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.compute.base import (
from libcloud.common.nttcis import (
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
def ex_edit_ip_address_list(self, ex_ip_address_list, description=None, ip_address_collection=None, child_ip_address_lists=None):
    """
        Edit IP Address List. IP Address list.
        Bear in mind you cannot add ip addresses to
        >>> from pprint import pprint
        >>> from libcloud.compute.types import Provider
        >>> from libcloud.compute.providers import get_driver
        >>> from libcloud.common.NTTCIS import NttCisIpAddress
        >>> import libcloud.security
        >>>
        >>> # Get NTTC-CIS driver
        >>> libcloud.security.VERIFY_SSL_CERT = True
        >>> cls = get_driver(Provider.NTTCIS)
        >>> driver = cls('myusername','mypassword', region='dd-au')
        >>>
        >>> # IP Address collection
        >>> ipAddress_1 = NttCisIpAddress(begin='190.2.2.100')
        >>> ipAddress_2 = NttCisIpAddress(begin='190.2.2.106',
        >>>                                      end='190.2.2.108')
        >>> ipAddress_3 = NttCisIpAddress(
        >>>                   begin='190.2.2.0', prefix_size='24')
        >>> ip_address_collection = [ipAddress_1, ipAddress_2, ipAddress_3]
        >>>
        >>> # Edit IP Address List
        >>> ip_address_list_id = '5e7c323f-c885-4e4b-9a27-94c44217dbd3'
        >>> result = driver.ex_edit_ip_address_list(
        >>>      ex_ip_address_list=ip_address_list_id,
        >>>      description="Edit Test",
        >>>      ip_address_collection=ip_address_collection,
        >>>      child_ip_address_lists=None
        >>>      )
        >>> pprint(result)

        :param    ex_ip_address_list:  (required) IpAddressList object or
                                       IpAddressList ID
        :type     ex_ip_address_list: :class:'NttCisIpAddressList'
                    or ``str``

        :param    description:  IP Address List Description
        :type      description: :``str``

        :param    ip_address_collection:  List of IP Address
        :type     ip_address_collection: ''list'' of
                                         :class:'NttCisIpAddressList'

        :param   child_ip_address_lists:  Child IP Address List or id to be
                                          included in this IP Address List
        :type    child_ip_address_lists:  ``list`` of
                                    :class:'NttCisChildIpAddressList'
                                    or ``str``

        :return: a list of NttCisIpAddressList objects
        :rtype: ``list`` of :class:`NttCisIpAddressList`
        """
    edit_ip_address_list = ET.Element('editIpAddressList', {'xmlns': TYPES_URN, 'id': self._ip_address_list_to_ip_address_list_id(ex_ip_address_list), 'xmlns:xsi': 'http://www.w3.org/2001/XMLSchema-instance'})
    if description is not None:
        if description != 'nil':
            ET.SubElement(edit_ip_address_list, 'description').text = description
        else:
            ET.SubElement(edit_ip_address_list, 'description', {'xsi:nil': 'true'})
    if ip_address_collection is not None:
        for ip in ip_address_collection:
            ip_address = ET.SubElement(edit_ip_address_list, 'ipAddress')
            ip_address.set('begin', ip.begin)
            if ip.end:
                ip_address.set('end', ip.end)
            if ip.prefix_size:
                ip_address.set('prefixSize', ip.prefix_size)
    if child_ip_address_lists is not None:
        ET.SubElement(edit_ip_address_list, 'childIpAddressListId').text = self._child_ip_address_list_to_child_ip_address_list_id(child_ip_address_lists)
    else:
        ET.SubElement(edit_ip_address_list, 'childIpAddressListId', {'xsi:nil': 'true'})
    response = self.connection.request_with_orgId_api_2('network/editIpAddressList', method='POST', data=ET.tostring(edit_ip_address_list)).object
    response_code = findtext(response, 'responseCode', TYPES_URN)
    return response_code in ['IN_PROGRESS', 'OK']