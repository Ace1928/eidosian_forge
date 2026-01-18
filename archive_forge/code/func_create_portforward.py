import json
from libcloud.compute.base import (
from libcloud.common.gig_g8 import G8Connection
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
def create_portforward(self, node, publicport, privateport, protocol='tcp'):
    return self.driver.ex_create_portforward(self, node, publicport, privateport, protocol)