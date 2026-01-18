import uuid
from boto.compat import urllib
from boto.resultset import ResultSet
def get_invalidation_request(self):
    """
        Returns an InvalidationBatch object representing the invalidation
        request referred to in the InvalidationSummary.

        :rtype: :class:`boto.cloudfront.invalidation.InvalidationBatch`
        :returns: An InvalidationBatch object representing the invalidation
                  request referred to by the InvalidationSummary
        """
    return self.connection.invalidation_request_status(self.distribution_id, self.id)