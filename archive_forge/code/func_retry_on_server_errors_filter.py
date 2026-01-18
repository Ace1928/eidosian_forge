import logging
import random
import sys
import time
import traceback
from google.cloud.ml.util import _exceptions
from six import reraise
def retry_on_server_errors_filter(exception):
    """Filter allowing retries on server errors and non-HttpErrors."""
    if isinstance(exception, _exceptions._RequestException):
        if exception.status >= 500:
            return True
        else:
            return False
    else:
        return True