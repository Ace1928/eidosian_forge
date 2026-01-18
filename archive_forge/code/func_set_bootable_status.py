from openstack.common import metadata
from openstack import format
from openstack import resource
from openstack import utils
def set_bootable_status(self, session, bootable=True):
    """Set volume bootable status flag"""
    body = {'os-set_bootable': {'bootable': bootable}}
    self._action(session, body)