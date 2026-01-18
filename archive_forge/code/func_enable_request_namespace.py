import os
import re
import warnings
from googlecloudsdk.third_party.appengine.api import lib_config
def enable_request_namespace():
    """Set the default namespace to the Google Apps domain referring this request.

  This method is deprecated, use lib_config instead.

  Calling this function will set the default namespace to the
  Google Apps domain that was used to create the url used for this request
  and only for the current request and only if the current default namespace
  is unset.

  """
    warnings.warn('namespace_manager.enable_request_namespace() is deprecated: use lib_config instead.', DeprecationWarning, stacklevel=2)
    if _ENV_CURRENT_NAMESPACE not in os.environ:
        if _ENV_DEFAULT_NAMESPACE in os.environ:
            os.environ[_ENV_CURRENT_NAMESPACE] = os.environ[_ENV_DEFAULT_NAMESPACE]