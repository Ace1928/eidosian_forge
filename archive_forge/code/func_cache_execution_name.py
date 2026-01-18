from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.workflows import cache
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
import six
def cache_execution_name(response, _):
    """Extracts the execution resource name to be saved into cache.

  Args:
    response: API response

  Returns:
    response: API response
  """
    cache.cache_execution_id(response.name)
    return response