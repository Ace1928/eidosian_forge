from __future__ import (absolute_import, division, print_function)
import time
def get_fixed_instance_size(oneandone_conn, fixed_instance_size, full_object=False):
    """
    Validates the fixed instance size exists by ID or name.
    Return the instance size ID.
    """
    for _fixed_instance_size in oneandone_conn.fixed_server_flavors():
        if fixed_instance_size in (_fixed_instance_size['id'], _fixed_instance_size['name']):
            if full_object:
                return _fixed_instance_size
            return _fixed_instance_size['id']