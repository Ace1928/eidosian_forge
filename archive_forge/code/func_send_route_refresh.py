import logging
import os
import netaddr
from . import docker_base as base
def send_route_refresh(self):
    self.vtysh('clear ip bgp * soft', config=False)