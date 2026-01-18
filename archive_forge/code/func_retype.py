from openstack.common import metadata
from openstack import format
from openstack import resource
from openstack import utils
def retype(self, session, new_type, migration_policy=None):
    """Change volume type"""
    body = {'os-retype': {'new_type': new_type}}
    if migration_policy:
        body['os-retype']['migration_policy'] = migration_policy
    self._action(session, body)