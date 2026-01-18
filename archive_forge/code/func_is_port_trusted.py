import random
import socket
import netaddr
from neutron_lib import constants
def is_port_trusted(port):
    """Used to determine if port can be trusted not to attack network.

    Trust is currently based on the device_owner field starting with 'network:'
    since we restrict who can use that in the default policy.yaml file.

    :param port: The port dict to inspect the 'device_owner' for.
    :returns: True if the port dict's 'device_owner' value starts with the
        networking prefix. False otherwise.
    """
    return port['device_owner'].startswith(constants.DEVICE_OWNER_NETWORK_PREFIX)