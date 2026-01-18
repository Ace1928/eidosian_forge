import logging
import random
from botocore.exceptions import (
from botocore.retries import quota, special
from botocore.retries.base import BaseRetryableChecker, BaseRetryBackoff
def create_retry_context(self, **kwargs):
    """Create context based on needs-retry kwargs."""
    response = kwargs['response']
    if response is None:
        http_response = None
        parsed_response = None
    else:
        http_response, parsed_response = response
    context = RetryContext(attempt_number=kwargs['attempts'], operation_model=kwargs['operation'], http_response=http_response, parsed_response=parsed_response, caught_exception=kwargs['caught_exception'], request_context=kwargs['request_dict']['context'])
    return context