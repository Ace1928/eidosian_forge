from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
import json
import os
from googlecloudsdk.core import config
from googlecloudsdk.core import log
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
def get_bin_folder():
    sdk_bin_path = config.Paths().sdk_bin_path
    if not sdk_bin_path:
        raise ECPConfigError('Unable to find the SDK bin path. The gcloud installation may be corrupted.')
    return sdk_bin_path