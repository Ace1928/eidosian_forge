from openstack.cloud import _utils
from openstack import exceptions
from openstack.orchestration.util import event_utils
from openstack.orchestration.v1._proxy import Proxy
def search_stacks(self, name_or_id=None, filters=None):
    """Search stacks.

        :param name_or_id: Name or ID of the desired stack.
        :param filters: a dict containing additional filters to use. e.g.
                {'stack_status': 'CREATE_COMPLETE'}

        :returns: a list of ``openstack.orchestration.v1.stack.Stack``
            containing the stack description.
        :raises: :class:`~openstack.exceptions.SDKException` if something goes
            wrong during the OpenStack API call.
        """
    stacks = self.list_stacks()
    return _utils._filter_list(stacks, name_or_id, filters)