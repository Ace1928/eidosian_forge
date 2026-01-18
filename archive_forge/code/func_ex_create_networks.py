import time
from libcloud.compute.base import (
from libcloud.utils.iso8601 import parse_date
from libcloud.common.gridscale import GridscaleBaseDriver, GridscaleConnection
from libcloud.compute.providers import Provider
def ex_create_networks(self, name, location):
    """
        Create a network at the data center location.

        :param name: Name of the network.
        :type name: ``str``

        :param location: Location.
        :type location: :class:`.NodeLocation`

        :return: Network.
        :rtype: :class:`.GridscaleNetwork`
        """
    self.connection.async_request('objects/networks', data={'name': name, 'location_uuid': location.id}, method='POST')
    return self._to_network(self._get_resource('network', self.connection.poll_response_initial.object['object_uuid']))