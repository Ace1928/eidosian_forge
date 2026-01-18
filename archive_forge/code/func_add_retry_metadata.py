import logging
import random
from botocore.exceptions import (
from botocore.retries import quota, special
from botocore.retries.base import BaseRetryableChecker, BaseRetryBackoff
def add_retry_metadata(self, **kwargs):
    """Add key/value pairs to the retry metadata.

        This allows any objects during the retry process to add
        metadata about any checks/validations that happened.

        This gets added to the response metadata in the retry handler.

        """
    self._retry_metadata.update(**kwargs)