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
def ip_route_delete(session, destination, **kwargs):
    """
    Deletes route record(s) from Zebra protocol service database.

    The arguments are similar to "ip route delete" command of iproute2.

    :param session: Session instance connecting to database.
    :param destination: Destination prefix.
    :param kwargs: Filtering rules to query.
    :return: Records which are deleted.
    """
    routes = ip_route_show_all(session, destination=destination, **kwargs)
    for route in routes:
        session.delete(route)
    return routes