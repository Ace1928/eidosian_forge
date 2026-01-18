from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import logging
import os
import sqlite3
import time
from typing import Dict
import uuid
import googlecloudsdk
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.configurations import named_configs
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files as file_utils
from googlecloudsdk.core.util import pkg_resources
from googlecloudsdk.core.util import platforms
import six
@property
def opt_in_prompting_cache_path(self):
    """Gets the path to the file to cache information about opt-in prompting.

    This is stored in the config directory instead of the installation state
    because if the SDK is installed as root, it will fail to persist the cache
    when you are running gcloud as a normal user.

    Returns:
      str, The path to the file.
    """
    return os.path.join(self.global_config_dir, '.last_opt_in_prompt.yaml')