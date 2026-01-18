from troveclient import base
from troveclient import common
from troveclient.v1 import users
def disable_instance_root(self, instance):
    """Implements root-disable for instances."""
    self._disable_root(self.instances_url % base.getid(instance))