from unittest import mock
from openstack.network.v2 import agent
from openstack.tests.unit import base
def test_add_agent_to_network(self):
    net = agent.Agent(**EXAMPLE)
    response = mock.Mock()
    response.body = {'network_id': '1'}
    response.json = mock.Mock(return_value=response.body)
    sess = mock.Mock()
    sess.post = mock.Mock(return_value=response)
    body = {'network_id': '1'}
    self.assertEqual(response.body, net.add_agent_to_network(sess, **body))
    url = 'agents/IDENTIFIER/dhcp-networks'
    sess.post.assert_called_with(url, json=body)