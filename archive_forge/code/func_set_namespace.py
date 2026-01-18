import os
import re
import warnings
from googlecloudsdk.third_party.appengine.api import lib_config
def set_namespace(namespace):
    """Set the default namespace for the current HTTP request.

  Args:
    namespace: A string naming the new namespace to use. A value of None
      will unset the default namespace value.
  """
    if namespace is None:
        os.environ.pop(_ENV_CURRENT_NAMESPACE, None)
    else:
        validate_namespace(namespace)
        os.environ[_ENV_CURRENT_NAMESPACE] = namespace