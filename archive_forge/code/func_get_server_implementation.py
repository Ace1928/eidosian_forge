from __future__ import (absolute_import, division, print_function)
from functools import reduce
import os
from ansible.module_utils.six.moves import configparser
from ansible.module_utils._text import to_native
from ansible_collections.community.mysql.plugins.module_utils.database import mysql_quote_identifier
def get_server_implementation(cursor):
    if 'mariadb' in get_server_version(cursor).lower():
        return 'mariadb'
    else:
        return 'mysql'