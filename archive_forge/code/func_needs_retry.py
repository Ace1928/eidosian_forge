import logging
import random
from botocore.exceptions import (
from botocore.retries import quota, special
from botocore.retries.base import BaseRetryableChecker, BaseRetryBackoff
def needs_retry(self, **kwargs):
    """Connect as a handler to the needs-retry event."""
    retry_delay = None
    context = self._retry_event_adapter.create_retry_context(**kwargs)
    if self._retry_policy.should_retry(context):
        if self._retry_quota.acquire_retry_quota(context):
            retry_delay = self._retry_policy.compute_retry_delay(context)
            logger.debug('Retry needed, retrying request after delay of: %s', retry_delay)
        else:
            logger.debug('Retry needed but retry quota reached, not retrying request.')
    else:
        logger.debug('Not retrying request.')
    self._retry_event_adapter.adapt_retry_response_from_context(context)
    return retry_delay