from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.cloudfront import CloudFrontConnection
from boto.cloudfront.distribution import Distribution, DistributionConfig, DistributionSummary
from boto.cloudfront.origin import CustomOrigin
def test_get_distribution_info(self):
    body = b'\n        <Distribution xmlns="http://cloudfront.amazonaws.com/doc/2010-11-01/">\n            <Id>EEEEEEEEEEEEE</Id>\n            <Status>InProgress</Status>\n            <LastModifiedTime>2014-02-03T11:03:41.087Z</LastModifiedTime>\n            <InProgressInvalidationBatches>0</InProgressInvalidationBatches>\n            <DomainName>abcdef12345678.cloudfront.net</DomainName>\n            <DistributionConfig>\n                <CustomOrigin>\n                    <DNSName>example.com</DNSName>\n                    <HTTPPort>80</HTTPPort>\n                    <HTTPSPort>443</HTTPSPort>\n                    <OriginProtocolPolicy>http-only</OriginProtocolPolicy>\n                </CustomOrigin>\n                <CallerReference>1111111111111</CallerReference>\n                <CNAME>static.example.com</CNAME>\n                <Enabled>true</Enabled>\n            </DistributionConfig>\n        </Distribution>\n        '
    self.set_http_response(status_code=200, body=body)
    response = self.service_connection.get_distribution_info('EEEEEEEEEEEEE')
    self.assertTrue(isinstance(response, Distribution))
    self.assertTrue(isinstance(response.config, DistributionConfig))
    self.assertTrue(isinstance(response.config.origin, CustomOrigin))
    self.assertEqual(response.config.origin.dns_name, 'example.com')
    self.assertEqual(response.config.origin.http_port, 80)
    self.assertEqual(response.config.origin.https_port, 443)
    self.assertEqual(response.config.origin.origin_protocol_policy, 'http-only')
    self.assertEqual(response.config.cnames, ['static.example.com'])
    self.assertTrue(response.config.enabled)
    self.assertEqual(response.id, 'EEEEEEEEEEEEE')
    self.assertEqual(response.status, 'InProgress')
    self.assertEqual(response.domain_name, 'abcdef12345678.cloudfront.net')
    self.assertEqual(response.in_progress_invalidation_batches, 0)