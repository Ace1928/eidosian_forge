import inspect
import json
import os
import pydoc  # used for importing python classes from their FQN
import sys
from ._interfaces import Model
from .prediction_utils import PredictionError
def load_custom_class():
    """Loads in the user specified custom class.

  Returns:
    An instance of a class specified by the user in the `create_version_request`
    or None if no such class was specified.

  Raises:
    PredictionError: if the user provided python class cannot be found.
  """
    create_version_json = os.environ.get('create_version_request')
    if not create_version_json:
        return None
    create_version_request = json.loads(create_version_json)
    if not create_version_request:
        return None
    version = create_version_request.get('version')
    if not version:
        return None
    class_name = version.get(_PREDICTION_CLASS_KEY)
    if not class_name:
        return None
    custom_class = pydoc.locate(class_name)
    if not custom_class:
        package_uris = [str(s) for s in version.get('package_uris')]
        raise PredictionError(PredictionError.INVALID_USER_CODE, '%s cannot be found. Please make sure (1) %s is the fully qualified function name, and (2) it uses the correct package name as provided by the package_uris: %s' % (class_name, _PREDICTION_CLASS_KEY, package_uris))
    return custom_class