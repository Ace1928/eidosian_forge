from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import enum
import hashlib
import json
import os
import re
from googlecloudsdk.command_lib.storage import encryption_util
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import hashing
from googlecloudsdk.core.util import platforms
from googlecloudsdk.core.util import scaled_integer
def raise_exceeds_max_length_error(file_name):
    if len(file_name) > _MAX_FILE_NAME_LENGTH:
        raise errors.Error('File name is over max character limit of {}: {}'.format(_MAX_FILE_NAME_LENGTH, file_name))