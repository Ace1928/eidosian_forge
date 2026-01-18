from __future__ import absolute_import, division, print_function
from datetime import timedelta
from decimal import Decimal
from os import environ
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.six import iteritems
from ansible_collections.community.postgresql.plugins.module_utils.version import \
def set_search_path(cursor, search_path):
    """Set session's search_path.

    Args:
        cursor (Psycopg cursor): Database cursor object.
        search_path (str): String containing comma-separated schema names.
    """
    cursor.execute('SET search_path TO %s' % search_path)