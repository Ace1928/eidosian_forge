import re
import sys
from libcloud.utils.py3 import ET, urlencode, basestring, ensure_string
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.compute.base import (
from libcloud.common.nttcis import (
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
def ex_edit_firewall_rule(self, rule, position=None, relative_rule_for_position=None):
    """
        Edit a firewall rule

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
        >>> # Get location
        >>> location = driver.ex_get_location_by_id(id='AU9')
        >>>
        >>> # Get network domain by location
        >>> networkDomainName = "Baas QA"
        >>> network_domains = driver.ex_list_network_domains(location=location)
        >>> my_network_domain = [d for d in network_domains if d.name ==
                              networkDomainName][0]
        >>>
        >>>
        >>> # List firewall rules
        >>> firewall_rules = driver.ex_list_firewall_rules(my_network_domain)
        >>>
        >>> # Get Firewall Rule by name
        >>> pprint("List specific firewall rule by name")
        >>> fire_rule_under_test = (list(filter(lambda x: x.name ==
                                   'My_New_Firewall_Rule', firewall_rules))[0])
        >>> pprint(fire_rule_under_test.source)
        >>> pprint(fire_rule_under_test.destination)
        >>>
        >>> # Edit Firewall
        >>> fire_rule_under_test.destination.address_list_id =
                '5e7c323f-c885-4e4b-9a27-94c44217dbd3'
        >>> fire_rule_under_test.destination.port_list_id =
                'b6557c5a-45fa-4138-89bd-8fe68392691b'
        >>> result = driver.ex_edit_firewall_rule(fire_rule_under_test, 'LAST')
        >>> pprint(result)

        :param rule: (required) The rule in which to create
        :type  rule: :class:`DNttCisFirewallRule`

        :param position: (required) There are two types of positions
                         with position_relative_to_rule arg and without it
                         With: 'BEFORE' or 'AFTER'
                         Without: 'FIRST' or 'LAST'
        :type  position: ``str``

        :param relative_rule_for_position: (optional) The rule or rule name in
                                           which to decide the relative rule
                                           for positioning.
        :type  relative_rule_for_position:
            :class:`NttCisFirewallRule` or ``str``

        :rtype: ``bool``
        """
    positions_without_rule = ('FIRST', 'LAST')
    positions_with_rule = ('BEFORE', 'AFTER')
    edit_node = ET.Element('editFirewallRule', {'xmlns': TYPES_URN, 'id': rule.id})
    ET.SubElement(edit_node, 'action').text = rule.action
    ET.SubElement(edit_node, 'protocol').text = rule.protocol
    source = ET.SubElement(edit_node, 'source')
    if rule.source.address_list_id is not None:
        source_ip = ET.SubElement(source, 'ipAddressListId')
        source_ip.text = rule.source.address_list_id
    else:
        source_ip = ET.SubElement(source, 'ip')
        if rule.source.any_ip:
            source_ip.set('address', 'ANY')
        else:
            source_ip.set('address', rule.source.ip_address)
            if rule.source.ip_prefix_size is not None:
                source_ip.set('prefixSize', str(rule.source.ip_prefix_size))
    if rule.source.port_list_id is not None:
        source_port = ET.SubElement(source, 'portListId')
        source_port.text = rule.source.port_list_id
    else:
        if rule.source.port_begin is not None:
            source_port = ET.SubElement(source, 'port')
            source_port.set('begin', rule.source.port_begin)
        if rule.source.port_end is not None:
            source_port.set('end', rule.source.port_end)
    dest = ET.SubElement(edit_node, 'destination')
    if rule.destination.address_list_id is not None:
        dest_ip = ET.SubElement(dest, 'ipAddressListId')
        dest_ip.text = rule.destination.address_list_id
    else:
        dest_ip = ET.SubElement(dest, 'ip')
        if rule.destination.any_ip:
            dest_ip.set('address', 'ANY')
        else:
            dest_ip.set('address', rule.destination.ip_address)
            if rule.destination.ip_prefix_size is not None:
                dest_ip.set('prefixSize', rule.destination.ip_prefix_size)
    if rule.destination.port_list_id is not None:
        dest_port = ET.SubElement(dest, 'portListId')
        dest_port.text = rule.destination.port_list_id
    else:
        if rule.destination.port_begin is not None:
            dest_port = ET.SubElement(dest, 'port')
            dest_port.set('begin', rule.destination.port_begin)
        if rule.destination.port_end is not None:
            dest_port.set('end', rule.destination.port_end)
    ET.SubElement(edit_node, 'enabled').text = str(rule.enabled).lower()
    if position is not None:
        placement = ET.SubElement(edit_node, 'placement')
        if relative_rule_for_position is not None:
            if position not in positions_with_rule:
                raise ValueError('When position_relative_to_rule is specified position must be %s' % ', '.join(positions_with_rule))
            if isinstance(relative_rule_for_position, NttCisFirewallRule):
                rule_name = relative_rule_for_position.name
            else:
                rule_name = relative_rule_for_position
            placement.set('relativeToRule', rule_name)
        elif position not in positions_without_rule:
            raise ValueError('When position_relative_to_rule is not specified position must be %s' % ', '.join(positions_without_rule))
        placement.set('position', position)
    response = self.connection.request_with_orgId_api_2('network/editFirewallRule', method='POST', data=ET.tostring(edit_node)).object
    response_code = findtext(response, 'responseCode', TYPES_URN)
    return response_code in ['IN_PROGRESS', 'OK']