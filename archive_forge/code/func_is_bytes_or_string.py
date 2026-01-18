from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import os
import re
import struct
import sys
import textwrap
import six
from six.moves import range  # pylint: disable=redefined-builtin
def is_bytes_or_string(maybe_string):
    if str is bytes:
        return isinstance(maybe_string, basestring)
    else:
        return isinstance(maybe_string, (str, bytes))