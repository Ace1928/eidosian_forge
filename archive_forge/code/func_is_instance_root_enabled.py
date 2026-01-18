from troveclient import base
from troveclient import common
from troveclient.v1 import users
def is_instance_root_enabled(self, instance):
    """Returns whether root is enabled for the instance."""
    return self._is_root_enabled(self.instances_url % base.getid(instance))