from openstack.common import metadata
from openstack import exceptions
from openstack import format
from openstack import resource
from openstack import utils
def init_attachment(self, session, connector):
    """Initialize volume attachment"""
    body = {'os-initialize_connection': {'connector': connector}}
    resp = self._action(session, body).json()
    return resp['connection_info']