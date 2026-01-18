import typing as ty
from openstack.common import metadata
from openstack.common import tag
from openstack.compute.v2 import flavor
from openstack.compute.v2 import volume_attachment
from openstack import exceptions
from openstack.image.v2 import image
from openstack import resource
from openstack import utils
def get_console_url(self, session, console_type):
    """Get the console URL for the server.

        :param session: The session to use for making this request.
        :param console_type: The type of console to return. This is
            cloud-specific. One of: ``novnc``, ``xvpvnc``, ``spice-html5``,
            ``rdp-html5``, ``serial``.
        :returns: None
        """
    action = CONSOLE_TYPE_ACTION_MAPPING.get(console_type)
    if not action:
        raise ValueError('Unsupported console type %s' % console_type)
    body = {action: {'type': console_type}}
    resp = self._action(session, body)
    return resp.json().get('console')