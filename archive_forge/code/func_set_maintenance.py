import collections
import enum
from openstack.baremetal.v1 import _common
from openstack import exceptions
from openstack import resource
from openstack import utils
def set_maintenance(self, session, reason=None):
    """Enable maintenance mode on the node.

        :param session: The session to use for making this request.
        :type session: :class:`~keystoneauth1.adapter.Adapter`
        :param reason: Optional reason for maintenance.
        :return: This :class:`Node` instance.
        """
    self._do_maintenance_action(session, 'put', {'reason': reason})
    return self.fetch(session)