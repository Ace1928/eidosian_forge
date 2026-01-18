from openstack.cloud import _utils
from openstack.cloud import exc
from openstack import exceptions
from openstack.network.v2._proxy import Proxy
def search_qos_policies(self, name_or_id=None, filters=None):
    """Search QoS policies

        :param name_or_id: Name or ID of the desired policy.
        :param filters: a dict containing additional filters to use. e.g.
            {'shared': True}

        :returns: A list of network ``QosPolicy`` objects matching the search
            criteria.
        :raises: :class:`~openstack.exceptions.SDKException` if something goes
            wrong during the OpenStack API call.
        """
    if not self._has_neutron_extension('qos'):
        raise exc.OpenStackCloudUnavailableExtension('QoS extension is not available on target cloud')
    query = {}
    if name_or_id:
        query['name'] = name_or_id
    if filters:
        query.update(filters)
    return list(self.network.qos_policies(**query))