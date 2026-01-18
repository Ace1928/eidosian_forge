import xml.sax
import time
import boto
from boto.connection import AWSAuthConnection
from boto import handler
from boto.cloudfront.distribution import Distribution, DistributionSummary, DistributionConfig
from boto.cloudfront.distribution import StreamingDistribution, StreamingDistributionSummary, StreamingDistributionConfig
from boto.cloudfront.identity import OriginAccessIdentity
from boto.cloudfront.identity import OriginAccessIdentitySummary
from boto.cloudfront.identity import OriginAccessIdentityConfig
from boto.cloudfront.invalidation import InvalidationBatch, InvalidationSummary, InvalidationListResultSet
from boto.resultset import ResultSet
from boto.cloudfront.exception import CloudFrontServerError
def invalidation_request_status(self, distribution_id, request_id, caller_reference=None):
    uri = '/%s/distribution/%s/invalidation/%s' % (self.Version, distribution_id, request_id)
    response = self.make_request('GET', uri, {'Content-Type': 'text/xml'})
    body = response.read()
    if response.status == 200:
        paths = InvalidationBatch([])
        h = handler.XmlHandler(paths, self)
        xml.sax.parseString(body, h)
        return paths
    else:
        raise CloudFrontServerError(response.status, response.reason, body)