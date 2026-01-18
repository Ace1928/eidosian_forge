from __future__ import (absolute_import, division, print_function)
import time
def get_datacenter(oneandone_conn, datacenter, full_object=False):
    """
    Validates the datacenter exists by ID or country code.
    Returns the datacenter ID.
    """
    for _datacenter in oneandone_conn.list_datacenters():
        if datacenter in (_datacenter['id'], _datacenter['country_code']):
            if full_object:
                return _datacenter
            return _datacenter['id']