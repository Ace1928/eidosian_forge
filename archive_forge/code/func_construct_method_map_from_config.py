from oslo_log import log
from pycadf import cadftaxonomy as taxonomy
from pycadf import reason
from pycadf import resource
from keystone.common import driver_hints
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone import notifications
def construct_method_map_from_config():
    """Determine authentication method types for deployment.

    :returns: a dictionary containing the methods and their indexes

    """
    method_map = dict()
    method_index = 1
    for method in CONF.auth.methods:
        method_map[method_index] = method
        method_index = method_index * 2
    return method_map