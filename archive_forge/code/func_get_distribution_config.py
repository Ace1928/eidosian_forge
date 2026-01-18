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
def get_distribution_config(self, distribution_id):
    return self._get_config(distribution_id, 'distribution', DistributionConfig)