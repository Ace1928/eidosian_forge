import logging
from sqlalchemy import Column
from sqlalchemy import Integer
from sqlalchemy import String
from os_ken.lib import netdevice
from os_ken.lib import ip
from os_ken.lib.packet import zebra
from . import base
@base.sql_function
def ip_link_show_all(session, **kwargs):
    """
    Returns all interface records matching the given filtering rules.

    The arguments for "kwargs" is the same with Interface class.

    :param session: Session instance connecting to database.
    :param kwargs: Filtering rules to query.
    :return: A list of Interface records.
    """
    return session.query(Interface).filter_by(**kwargs).all()