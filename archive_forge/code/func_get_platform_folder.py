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
def get_platform_folder():
    sdk_root = config.Paths().sdk_root
    if not sdk_root:
        raise ECPConfigError('Unable to find the SDK root path. The gcloud installation may be corrupted.')
    return os.path.join(sdk_root, 'platform', 'enterprise_cert')