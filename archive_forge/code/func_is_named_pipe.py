from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import enum
import os
import re
import stat
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.core import log
from googlecloudsdk.core.util import platforms
import six
from six.moves import urllib
def is_named_pipe(path):
    return os.path.exists(path) and stat.S_ISFIFO(os.stat(path).st_mode)