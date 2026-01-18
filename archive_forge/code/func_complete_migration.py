from openstack.common import metadata
from openstack import format
from openstack import resource
from openstack import utils
def complete_migration(self, session, new_volume_id, error=False):
    """Complete volume migration"""
    body = {'os-migrate_volume_completion': {'new_volume': new_volume_id, 'error': error}}
    self._action(session, body)