from openstack import exceptions
from openstack import resource
from openstack import utils
def add_host(self, session, host):
    """Adds a host to an aggregate."""
    body = {'add_host': {'host': host}}
    return self._action(session, body)