from openstack import exceptions
from openstack import resource
from openstack import utils
def force_complete(self, session):
    """Force on-going live migration to complete."""
    body = {'force_complete': None}
    self._action(session, body)