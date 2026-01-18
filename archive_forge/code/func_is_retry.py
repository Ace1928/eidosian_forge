from __future__ import absolute_import
import email
import logging
import re
import time
import warnings
from collections import namedtuple
from itertools import takewhile
from ..exceptions import (
from ..packages import six
def is_retry(self, method, status_code, has_retry_after=False):
    """Is this method/status code retryable? (Based on allowlists and control
        variables such as the number of total retries to allow, whether to
        respect the Retry-After header, whether this header is present, and
        whether the returned status code is on the list of status codes to
        be retried upon on the presence of the aforementioned header)
        """
    if not self._is_method_retryable(method):
        return False
    if self.status_forcelist and status_code in self.status_forcelist:
        return True
    return self.total and self.respect_retry_after_header and has_retry_after and (status_code in self.RETRY_AFTER_STATUS_CODES)