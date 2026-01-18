from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import binascii
import codecs
import os
import sys
import io
import re
import locale
import collections
import random
import six
import string
from six.moves import urllib
from six.moves import range
from gslib.exception import CommandException
from gslib.lazy_wrapper import LazyWrapper
from gslib.utils.constants import UTF8
from gslib.utils.constants import WINDOWS_1252
from gslib.utils.system_util import IS_CP1252
def get_random_ascii_chars(size, seed=0):
    """Generates binary string representation of a list of ASCII characters.

  Args:
    size: Integer quantity of characters to generate.
    seed: A seed may be specified for deterministic behavior.
          Int 0 is used as the default value.

  Returns:
    Binary encoded string representation of a list of characters of length
    equal to size argument.
  """
    random.seed(seed)
    contents = str([random.choice(string.ascii_letters) for _ in range(size)])
    contents = six.ensure_binary(contents)
    random.seed()
    return contents