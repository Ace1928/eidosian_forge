from __future__ import (absolute_import, division, print_function)
from functools import reduce
import os
from ansible.module_utils.six.moves import configparser
from ansible.module_utils._text import to_native
from ansible_collections.community.mysql.plugins.module_utils.database import mysql_quote_identifier
def get_connector_version(connector):
    """ (class) -> str
    Return the version of pymysql or mysqlclient (MySQLdb).
    Return 'Unknown' if the connector name is unknown.
    """
    if connector is None:
        return 'Unknown'
    connector_name = get_connector_name(connector)
    if connector_name == 'pymysql':
        v = connector.VERSION[:3]
        return '.'.join(map(str, v))
    elif connector_name == 'MySQLdb':
        v = connector.version_info[:3]
        return '.'.join(map(str, v))
    else:
        return 'Unknown'