import os
import subprocess
import six
from google.auth import _helpers
from google.auth import environment_vars
from google.auth import exceptions
def get_application_default_credentials_path():
    """Gets the path to the application default credentials file.

    The path may or may not exist.

    Returns:
        str: The full path to application default credentials.
    """
    config_path = get_config_path()
    return os.path.join(config_path, _CREDENTIALS_FILENAME)