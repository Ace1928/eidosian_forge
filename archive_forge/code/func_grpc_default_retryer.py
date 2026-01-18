from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import functools
import sys
from google.auth import exceptions as auth_exceptions
from googlecloudsdk.api_lib.storage import errors
from googlecloudsdk.api_lib.storage import retry_util
import requests
def grpc_default_retryer(func):
    """A decorator to retry on transient errors."""

    @functools.wraps(func)
    def wrapped_func(*args, **kwargs):
        return retry_util.retryer(target=func, should_retry_if=is_retriable, target_args=args, target_kwargs=kwargs)
    return wrapped_func