from builtins import str  # remove this once Py2 is dropped
import json
import os
from . import _version
def is_ci():
    """
    Returns a boolean. Will be `True` if the code is running on a CI server,
    otherwise `False`.
    """
    return bool(name())