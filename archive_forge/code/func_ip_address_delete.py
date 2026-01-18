import logging
from sqlalchemy import Column
from sqlalchemy import Integer
from sqlalchemy import String
from os_ken.lib import netdevice
from os_ken.lib import ip
from os_ken.lib.packet import zebra
from . import base
@base.sql_function
def ip_address_delete(session, ifname, ifaddr):
    """
    Deletes an IP address from interface record identified with the given
    "ifname".

    The arguments are similar to "ip address delete" command of iproute2.

    :param session: Session instance connecting to database.
    :param ifname: Name of interface.
    :param ifaddr: IPv4 or IPv6 address.
    :return: Instance of record or "None" if failed.
    """

    def _remove_inet_addr(intf_inet, addr):
        addr_list = intf_inet.split(',')
        if addr not in addr_list:
            LOG.debug('Interface "%s" does not have "ifaddr": %s', intf.ifname, addr)
            return intf_inet
        else:
            addr_list.remove(addr)
            return ','.join(addr_list)
    intf = ip_link_show(session, ifname=ifname)
    if not intf:
        LOG.debug('Interface "%s" does not exist', ifname)
        return None
    if ip.valid_ipv4(ifaddr):
        intf.inet = _remove_inet_addr(intf.inet, ifaddr)
    elif ip.valid_ipv6(ifaddr):
        intf.inet6 = _remove_inet_addr(intf.inet6, ifaddr)
    else:
        LOG.debug('Invalid IP address for "ifaddr": %s', ifaddr)
        return None
    return intf