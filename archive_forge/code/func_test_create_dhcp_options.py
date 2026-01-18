from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.vpc import VPCConnection, DhcpOptions
def test_create_dhcp_options(self):
    self.set_http_response(status_code=200)
    api_response = self.service_connection.create_dhcp_options(domain_name='example.com', domain_name_servers=['10.2.5.1', '10.2.5.2'], ntp_servers=('10.12.12.1', '10.12.12.2'), netbios_name_servers='10.20.20.1', netbios_node_type='2')
    self.assert_request_parameters({'Action': 'CreateDhcpOptions', 'DhcpConfiguration.1.Key': 'domain-name', 'DhcpConfiguration.1.Value.1': 'example.com', 'DhcpConfiguration.2.Key': 'domain-name-servers', 'DhcpConfiguration.2.Value.1': '10.2.5.1', 'DhcpConfiguration.2.Value.2': '10.2.5.2', 'DhcpConfiguration.3.Key': 'ntp-servers', 'DhcpConfiguration.3.Value.1': '10.12.12.1', 'DhcpConfiguration.3.Value.2': '10.12.12.2', 'DhcpConfiguration.4.Key': 'netbios-name-servers', 'DhcpConfiguration.4.Value.1': '10.20.20.1', 'DhcpConfiguration.5.Key': 'netbios-node-type', 'DhcpConfiguration.5.Value.1': '2'}, ignore_params_values=['AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp', 'Version'])
    self.assertIsInstance(api_response, DhcpOptions)
    self.assertEquals(api_response.id, 'dopt-7a8b9c2d')
    self.assertEquals(api_response.options['domain-name'], ['example.com'])
    self.assertEquals(api_response.options['domain-name-servers'], ['10.2.5.1', '10.2.5.2'])
    self.assertEquals(api_response.options['ntp-servers'], ['10.12.12.1', '10.12.12.2'])
    self.assertEquals(api_response.options['netbios-name-servers'], ['10.20.20.1'])
    self.assertEquals(api_response.options['netbios-node-type'], ['2'])