from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.workflows import cache
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
import six
def print_default_location_warning(_, args, request):
    """Prints a warning when the default location is used.

  Args:
    args: gcloud command arguments
    request: API request

  Returns:
    request: API request
  """
    if not (properties.VALUES.workflows.location.IsExplicitlySet() or args.IsSpecified('location')):
        log.warning('The default location(us-central1) was used since the location flag was not specified.')
    return request