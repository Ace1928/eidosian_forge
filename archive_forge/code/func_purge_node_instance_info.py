import contextlib
import sys
import warnings
import jsonpatch
from openstack.baremetal.v1._proxy import Proxy
from openstack import exceptions
from openstack import warnings as os_warnings
def purge_node_instance_info(self, uuid):
    warnings.warn('The purge_node_instance_info call is deprecated, use patch_machine or update_machine instead', os_warnings.OpenStackDeprecationWarning)
    return self.patch_machine(uuid, dict(path='/instance_info', op='remove'))