from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.storage import errors as cloud_errors
def catch_http_error_raise_gcs_api_error(format_str=None):
    """Decorator catches HttpError and returns GcsApiError with custom message.

  Args:
    format_str (str): A googlecloudsdk.api_lib.util.exceptions.HttpErrorPayload
      format string. Note that any properties that are accessed here are on the
      HttpErrorPayload object, not the object returned from the server.

  Returns:
    A decorator that catches apitools.HttpError and returns GcsApiError with a
      customizable error message.
  """
    return cloud_errors.catch_error_raise_cloud_api_error(ERROR_TRANSLATION, format_str=format_str, status_code_getter=get_status_code)