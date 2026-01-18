from __future__ import absolute_import, division, print_function
from datetime import timedelta
from decimal import Decimal
from os import environ
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.six import iteritems
from ansible_collections.community.postgresql.plugins.module_utils.version import \
def exec_sql(obj, query, query_params=None, return_bool=False, add_to_executed=True, dont_exec=False):
    """Execute SQL.

    Auxiliary function for PostgreSQL user classes.

    Returns a query result if possible or a boolean value.

    Args:
        obj (obj) -- must be an object of a user class.
            The object must have module (AnsibleModule class object) and
            cursor (psycopg cursor object) attributes
        query (str) -- SQL query to execute

    Kwargs:
        query_params (dict or tuple) -- Query parameters to prevent SQL injections,
            could be a dict or tuple
        return_bool (bool) -- return True instead of rows if a query was successfully executed.
            It's necessary for statements that don't return any result like DDL queries (default False).
        add_to_executed (bool) -- append the query to obj.executed_queries attribute
        dont_exec (bool) -- used with add_to_executed=True to generate a query, add it
            to obj.executed_queries list and return True (default False)
    """
    if dont_exec:
        query = obj.cursor.mogrify(query, query_params)
        if add_to_executed:
            obj.executed_queries.append(query)
        return True
    try:
        if query_params is not None:
            obj.cursor.execute(query, query_params)
        else:
            obj.cursor.execute(query)
        if add_to_executed:
            if query_params is not None:
                obj.executed_queries.append(obj.cursor.mogrify(query, query_params))
            else:
                obj.executed_queries.append(query)
        if not return_bool:
            res = obj.cursor.fetchall()
            return res
        return True
    except Exception as e:
        obj.module.fail_json(msg="Cannot execute SQL '%s': %s" % (query, to_native(e)))
    return False