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
def installation_properties_path(self):
    """Gets the path to the installation-wide properties file.

    Returns:
      str, The path to the file.
    """
    sdk_root = self.sdk_root
    if not sdk_root:
        return None
    return os.path.join(sdk_root, self.CLOUDSDK_PROPERTIES_NAME)