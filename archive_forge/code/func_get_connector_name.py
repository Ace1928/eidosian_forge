from __future__ import (absolute_import, division, print_function)
from functools import reduce
import os
from ansible.module_utils.six.moves import configparser
from ansible.module_utils._text import to_native
from ansible_collections.community.mysql.plugins.module_utils.database import mysql_quote_identifier
def get_connector_name(connector):
    """ (class) -> str
    Return the name of the connector (pymysql or mysqlclient (MySQLdb))
    or 'Unknown' if not pymysql or MySQLdb. When adding a
    connector here, also modify get_connector_version.
    """
    if connector is None or not hasattr(connector, '__name__'):
        return 'Unknown'
    return connector.__name__