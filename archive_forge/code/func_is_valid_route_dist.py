import numbers
import re
import socket
from os_ken.lib import ip
def is_valid_route_dist(route_dist):
    """Validates *route_dist* as string representation of route distinguisher.

    Returns True if *route_dist* is as per our convention of RD, else False.
    Our convention is to represent RD as a string in format:
    *admin_sub_field:assigned_num_field* and *admin_sub_field* can be valid
    IPv4 string representation.
    Valid examples: '65000:222', '1.2.3.4:4432'.
    Invalid examples: '1.11.1: 333'
    """
    return is_valid_ext_comm_attr(route_dist)