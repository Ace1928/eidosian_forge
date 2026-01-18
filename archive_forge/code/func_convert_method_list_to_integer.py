from oslo_log import log
from pycadf import cadftaxonomy as taxonomy
from pycadf import reason
from pycadf import resource
from keystone.common import driver_hints
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone import notifications
def convert_method_list_to_integer(methods):
    """Convert the method type(s) to an integer.

    :param methods: a list of method names
    :returns: an integer representing the methods

    """
    method_map = construct_method_map_from_config()
    method_ints = []
    for method in methods:
        for k, v in method_map.items():
            if v == method:
                method_ints.append(k)
    return sum(method_ints)