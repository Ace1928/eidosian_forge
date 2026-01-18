from __future__ import (absolute_import, division, print_function)
import time
def get_private_network(oneandone_conn, private_network, full_object=False):
    """
    Validates the private network exists by ID or name.
    Return the private network ID.
    """
    for _private_network in oneandone_conn.list_private_networks():
        if private_network in (_private_network['name'], _private_network['id']):
            if full_object:
                return _private_network
            return _private_network['id']