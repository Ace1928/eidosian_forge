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
def ip_route_show_all(session, **kwargs):
    """
    Returns a selected route record matching the given filtering rules.

    The arguments are similar to "ip route showdump" command of iproute2.

    If "is_selected=True", disables the existing selected route for the
    given destination.

    :param session: Session instance connecting to database.
    :param kwargs: Filtering rules to query.
    :return: A list of route records.
    """
    return session.query(Route).filter_by(**kwargs).all()