import collections
import logging
import os_ken.exception as os_ken_exc
from os_ken.base import app_manager
from os_ken.controller import event
def register_key(self, network_id, tunnel_key):
    self.tunnel_keys.register_key(network_id, tunnel_key)