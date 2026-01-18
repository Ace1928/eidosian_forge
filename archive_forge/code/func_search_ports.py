from openstack.cloud import _utils
from openstack.cloud import exc
from openstack import exceptions
from openstack.network.v2._proxy import Proxy
def search_ports(self, name_or_id=None, filters=None):
    """Search ports

        :param name_or_id: Name or ID of the desired port.
        :param filters: A dict containing additional filters to use. e.g.
            {'device_id': '2711c67a-b4a7-43dd-ace7-6187b791c3f0'}

        :returns: A list of network ``Port`` objects matching the search
            criteria.
        :raises: :class:`~openstack.exceptions.SDKException` if something goes
            wrong during the OpenStack API call.
        """
    if isinstance(filters, str):
        pushdown_filters = None
    else:
        pushdown_filters = filters
    ports = self.list_ports(pushdown_filters)
    return _utils._filter_list(ports, name_or_id, filters)