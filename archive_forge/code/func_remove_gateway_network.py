from openstack import exceptions
from openstack import resource
from openstack import utils
def remove_gateway_network(self, session, network_id):
    """Delete Network from a BGP Speaker

        :param session: The session to communicate through.
        :type session: :class:`~keystoneauth1.adapter.Adapter`
        :param network_id: The ID of the network to disassociate
               from the speaker
        """
    body = {'network_id': network_id}
    url = utils.urljoin(self.base_path, self.id, 'remove_gateway_network')
    session.put(url, json=body)