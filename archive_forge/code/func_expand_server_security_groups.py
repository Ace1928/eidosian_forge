import ipaddress
import socket
from openstack import _log
from openstack import exceptions
from openstack import utils
def expand_server_security_groups(cloud, server):
    try:
        groups = cloud.list_server_security_groups(server)
    except exceptions.SDKException:
        groups = []
    server['security_groups'] = groups or []