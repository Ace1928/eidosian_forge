import ipaddress
import socket
from openstack import _log
from openstack import exceptions
from openstack import utils
def obj_list_to_munch(obj_list):
    """Enumerate through lists of objects and return lists of dictonaries.

    Some of the objects returned in OpenStack are actually lists of objects,
    and in order to expose the data structures as JSON, we need to facilitate
    the conversion to lists of dictonaries.
    """
    return [obj_to_munch(obj) for obj in obj_list]