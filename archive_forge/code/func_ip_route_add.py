import logging
import socket
from sqlalchemy import Column
from sqlalchemy import Boolean
from sqlalchemy import Integer
from sqlalchemy import String
from os_ken.lib import ip
from os_ken.lib.packet import safi as packet_safi
from os_ken.lib.packet import zebra
from . import base
from . import interface
@base.sql_function
def ip_route_add(session, destination, device=None, gateway='', source='', ifindex=0, route_type=zebra.ZEBRA_ROUTE_KERNEL, is_selected=True):
    """
    Adds a route record into Zebra protocol service database.

    The arguments are similar to "ip route add" command of iproute2.

    If "is_selected=True", disables the existing selected route for the
    given destination.

    :param session: Session instance connecting to database.
    :param destination: Destination prefix.
    :param device: Source device.
    :param gateway: Gateway IP address.
    :param source: Source IP address.
    :param ifindex: Index of source device.
    :param route_type: Route type of daemon (or kernel).
    :param is_selected: If select the given route as "in use" or not.
    :return: Instance of record or "None" if failed.
    """
    if device:
        intf = interface.ip_link_show(session, ifname=device)
        if not intf:
            LOG.debug('Interface "%s" does not exist', device)
            return None
        ifindex = ifindex or intf.ifindex
        route = ip_route_show(session, destination=destination, device=device)
        if route:
            LOG.debug('Route to "%s" already exists on "%s" device', destination, device)
            return route
    dest_addr, dest_prefix_num = destination.split('/')
    dest_prefix_num = int(dest_prefix_num)
    if ip.valid_ipv4(dest_addr) and 0 <= dest_prefix_num <= 32:
        family = socket.AF_INET
    elif ip.valid_ipv6(dest_addr) and 0 <= dest_prefix_num <= 128:
        family = socket.AF_INET6
    else:
        LOG.debug('Invalid IP address for "prefix": %s', destination)
        return None
    safi = packet_safi.UNICAST
    if is_selected:
        old_routes = ip_route_show_all(session, destination=destination, is_selected=True)
        for old_route in old_routes:
            if old_route:
                LOG.debug('Set existing route to unselected: %s', old_route)
                old_route.is_selected = False
    new_route = Route(family=family, safi=safi, destination=destination, gateway=gateway, ifindex=ifindex, source=source, route_type=route_type, is_selected=is_selected)
    session.add(new_route)
    return new_route