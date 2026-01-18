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
def sdk_bin_path(self):
    """Forms a path to bin directory by using sdk_root.

    Returns:
      str, The path to the bin directory of the Cloud SDK or None if it could
      not be found.
    """
    sdk_root = self.sdk_root
    return os.path.join(sdk_root, 'bin') if sdk_root else None