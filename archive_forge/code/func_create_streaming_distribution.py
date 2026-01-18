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
def create_streaming_distribution(self, origin, enabled, caller_reference='', cnames=None, comment='', trusted_signers=None):
    config = StreamingDistributionConfig(origin=origin, enabled=enabled, caller_reference=caller_reference, cnames=cnames, comment=comment, trusted_signers=trusted_signers)
    return self._create_object(config, 'streaming-distribution', StreamingDistribution)