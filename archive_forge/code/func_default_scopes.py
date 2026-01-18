import abc
import os
import six
from google.auth import _helpers, environment_vars
from google.auth import exceptions
@property
def default_scopes(self):
    """Sequence[str]: the credentials' current set of default scopes."""
    return self._default_scopes