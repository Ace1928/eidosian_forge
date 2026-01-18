import logging
from sqlalchemy import Column
from sqlalchemy import Integer
from sqlalchemy import String
from os_ken.lib import netdevice
from os_ken.lib import ip
from os_ken.lib.packet import zebra
from . import base
@base.sql_function
def ip_link_delete(session, name):
    """
    Deletes an interface record from Zebra protocol service database.

    The arguments are similar to "ip link delete" command of iproute2.

    :param session: Session instance connecting to database.
    :param name: Name of interface.
    :return: Name of interface which was deleted. None if failed.
    """
    intf = ip_link_show(session, ifname=name)
    if not intf:
        LOG.debug('Interface "%s" does not exist', name)
        return None
    session.delete(intf)
    return name