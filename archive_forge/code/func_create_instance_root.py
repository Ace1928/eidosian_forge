from troveclient import base
from troveclient import common
from troveclient.v1 import users
def create_instance_root(self, instance, root_password=None):
    """Implements root-enable for instances."""
    return self._enable_root(self.instances_url % base.getid(instance), root_password)