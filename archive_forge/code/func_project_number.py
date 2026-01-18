from the current environment without the need to copy, save and manage
import abc
import copy
import datetime
import io
import json
import re
from google.auth import _helpers
from google.auth import credentials
from google.auth import exceptions
from google.auth import impersonated_credentials
from google.auth import metrics
from google.oauth2 import sts
from google.oauth2 import utils
@property
def project_number(self):
    """Optional[str]: The project number corresponding to the workload identity pool."""
    components = self._audience.split('/')
    try:
        project_index = components.index('projects')
        if project_index + 1 < len(components):
            return components[project_index + 1] or None
    except ValueError:
        return None