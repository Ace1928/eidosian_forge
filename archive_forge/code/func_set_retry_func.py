from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import errno
from apitools.base.py import http_wrapper as apitools_http_wrapper
from googlecloudsdk.api_lib.storage import errors
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import retry
def set_retry_func(apitools_transfer_object):
    """Sets the retry function for the apitools transfer object.

  Replaces the Apitools' default retry function
  HandleExceptionsAndRebuildHttpConnections with a custom one which calls
  HandleExceptionsAndRebuildHttpConnections and then raise a custom exception.
  This is useful when we don't want MakeRequest method in Apitools to retry
  the http request directly and instead let the caller decide the next action.

  Args:
    apitools_transfer_object (apitools.base.py.transfer.Transfer): The
    Apitools' transfer object.
  """

    def _handle_error_and_raise(retry_args):
        apitools_http_wrapper.HandleExceptionsAndRebuildHttpConnections(retry_args)
        if isinstance(retry_args.exc, OSError) and retry_args.exc.errno == errno.ENOSPC:
            raise retry_args.exc
        raise errors.RetryableApiError()
    apitools_transfer_object.retry_func = _handle_error_and_raise