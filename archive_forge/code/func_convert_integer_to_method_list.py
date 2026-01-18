from oslo_log import log
from pycadf import cadftaxonomy as taxonomy
from pycadf import reason
from pycadf import resource
from keystone.common import driver_hints
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone import notifications
def convert_integer_to_method_list(method_int):
    """Convert an integer to a list of methods.

    :param method_int: an integer representing methods
    :returns: a corresponding list of methods

    """
    if method_int == 0:
        return []
    method_map = construct_method_map_from_config()
    method_ints = sorted(method_map, reverse=True)
    methods = []
    for m_int in method_ints:
        result = int(method_int / m_int)
        if result == 1:
            methods.append(method_map[m_int])
            method_int = method_int - m_int
    return methods