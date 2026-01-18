from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.cloudfront import CloudFrontConnection
from boto.cloudfront.distribution import Distribution, DistributionConfig, DistributionSummary
from boto.cloudfront.origin import CustomOrigin
def test_get_distribution_config(self):
    body = b'\n        <DistributionConfig xmlns="http://cloudfront.amazonaws.com/doc/2010-11-01/">\n        <CustomOrigin>\n            <DNSName>example.com</DNSName>\n            <HTTPPort>80</HTTPPort>\n            <HTTPSPort>443</HTTPSPort>\n            <OriginProtocolPolicy>http-only</OriginProtocolPolicy>\n        </CustomOrigin>\n        <CallerReference>1234567890123</CallerReference>\n        <CNAME>static.example.com</CNAME>\n        <Enabled>true</Enabled>\n        </DistributionConfig>\n        '
    self.set_http_response(status_code=200, body=body, header={'Etag': 'AABBCC'})
    response = self.service_connection.get_distribution_config('EEEEEEEEEEEEE')
    self.assertTrue(isinstance(response, DistributionConfig))
    self.assertTrue(isinstance(response.origin, CustomOrigin))
    self.assertEqual(response.origin.dns_name, 'example.com')
    self.assertEqual(response.origin.http_port, 80)
    self.assertEqual(response.origin.https_port, 443)
    self.assertEqual(response.origin.origin_protocol_policy, 'http-only')
    self.assertEqual(response.cnames, ['static.example.com'])
    self.assertTrue(response.enabled)
    self.assertEqual(response.etag, 'AABBCC')