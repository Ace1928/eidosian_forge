from openstack import exceptions
from openstack import resource
from openstack import utils
def remove_bgp_speaker_from_dragent(self, session, bgp_agent_id):
    """Delete BGP Speaker from a Dynamic Routing Agent

        :param session: The session to communicate through.
        :type session: :class:`~keystoneauth1.adapter.Adapter`
        :param bgp_agent_id: The id of the dynamic routing agent from which
                             remove the speaker.
        """
    url = utils.urljoin('agents', bgp_agent_id, 'bgp-drinstances', self.id)
    session.delete(url)