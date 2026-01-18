from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import contextlib
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
from six import text_type
from six.moves.urllib import parse
@contextlib.contextmanager
def override_endpoint(location):
    """Set api_endpoint_overrides property to use the regional endpoint.

  Args:
    location: The location used for the regional endpoint. (optional)

  Yields:
    None
  """
    old_endpoint = apis.GetEffectiveApiEndpoint('securedlandingzone', 'v1beta')
    try:
        if location != 'global':
            regional_endpoint = derive_regional_endpoint(old_endpoint, location)
            properties.VALUES.api_endpoint_overrides.securedlandingzone.Set(regional_endpoint)
        yield
    finally:
        properties.VALUES.api_endpoint_overrides.securedlandingzone.Set(old_endpoint)