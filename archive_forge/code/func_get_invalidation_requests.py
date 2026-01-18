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
def get_invalidation_requests(self, distribution_id, marker=None, max_items=None):
    """
        Get all invalidation requests for a given CloudFront distribution.
        This returns an instance of an InvalidationListResultSet that
        automatically handles all of the result paging, etc. from CF - you just
        need to keep iterating until there are no more results.

        :type distribution_id: string
        :param distribution_id: The id of the CloudFront distribution

        :type marker: string
        :param marker: Use this only when paginating results and only in
                       follow-up request after you've received a response where
                       the results are truncated. Set this to the value of the
                       Marker element in the response you just received.

        :type max_items: int
        :param max_items: Use this only when paginating results and only in a
                          follow-up request to indicate the maximum number of
                          invalidation requests you want in the response. You
                          will need to pass the next_marker property from the
                          previous InvalidationListResultSet response in the
                          follow-up request in order to get the next 'page' of
                          results.

        :rtype: :class:`boto.cloudfront.invalidation.InvalidationListResultSet`
        :returns: An InvalidationListResultSet iterator that lists invalidation
                  requests for a given CloudFront distribution. Automatically
                  handles paging the results.
        """
    uri = 'distribution/%s/invalidation' % distribution_id
    params = dict()
    if marker:
        params['Marker'] = marker
    if max_items:
        params['MaxItems'] = max_items
    if params:
        uri += '?%s=%s' % params.popitem()
        for k, v in params.items():
            uri += '&%s=%s' % (k, v)
    tags = [('InvalidationSummary', InvalidationSummary)]
    rs_class = InvalidationListResultSet
    rs_kwargs = dict(connection=self, distribution_id=distribution_id, max_items=max_items, marker=marker)
    return self._get_all_objects(uri, tags, result_set_class=rs_class, result_set_kwargs=rs_kwargs)