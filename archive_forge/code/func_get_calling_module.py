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
def get_calling_module():
    """Returns the name of the module that's calling into this module."""
    return get_calling_module_object_and_name().module_name