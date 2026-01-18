import re
import sys
from libcloud.utils.py3 import ET, urlencode, basestring, ensure_string
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.compute.base import (
from libcloud.common.nttcis import (
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
def ex_list_nodes_paginated(self, name=None, location=None, ipv6=None, ipv4=None, vlan=None, image=None, deployed=None, started=None, state=None, network=None, network_domain=None):
    """
        Return a generator which yields node lists in pages

        :keyword location: Filters the node list to nodes that are
                           located in this location
        :type    location: :class:`NodeLocation` or ``str``

        :keyword name: Filters the node list to nodes that have this name
        :type    name ``str``

        :keyword ipv6: Filters the node list to nodes that have this
                       ipv6 address
        :type    ipv6: ``str``

        :keyword ipv4: Filters the node list to nodes that have this
                       ipv4 address
        :type    ipv4: ``str``

        :keyword vlan: Filters the node list to nodes that are in this VLAN
        :type    vlan: :class:`NttCisVlan` or ``str``

        :keyword image: Filters the node list to nodes that have this image
        :type    image: :class:`NodeImage` or ``str``

        :keyword deployed: Filters the node list to nodes that are
                           deployed or not
        :type    deployed: ``bool``

        :keyword started: Filters the node list to nodes that are
                          started or not
        :type    started: ``bool``
        :keyword state: Filters the node list to nodes that are in
                        this state
        :type    state: ``str``
        :keyword network: Filters the node list to nodes in this network
        :type    network: :class:`NttCisNetwork` or ``str``

        :keyword network_domain: Filters the node list to nodes in this
                                 network domain
        :type    network_domain: :class:`NttCisNetworkDomain`
                                 or ``str``
        :return: a list of `Node` objects
        :rtype: ``generator`` of `list` of :class:`Node`

        """
    params = {}
    if location is not None:
        params['datacenterId'] = self._location_to_location_id(location)
    if ipv6 is not None:
        params['ipv6'] = ipv6
    if ipv4 is not None:
        params['privateIpv4'] = ipv4
    if state is not None:
        params['state'] = state
    if started is not None:
        params['started'] = started
    if deployed is not None:
        params['deployed'] = deployed
    if name is not None:
        params['name'] = name
    if network_domain is not None:
        params['networkDomainId'] = self._network_domain_to_network_domain_id(network_domain)
    if network is not None:
        params['networkId'] = self._network_to_network_id(network)
    if vlan is not None:
        params['vlanId'] = self._vlan_to_vlan_id(vlan)
    if image is not None:
        params['sourceImageId'] = self._image_to_image_id(image)
    nodes_obj = self._list_nodes_single_page(params)
    yield self._to_nodes(nodes_obj)
    while nodes_obj.get('pageCount') >= nodes_obj.get('pageSize'):
        params['pageNumber'] = int(nodes_obj.get('pageNumber')) + 1
        nodes_obj = self._list_nodes_single_page(params)
        yield self._to_nodes(nodes_obj)